import json
import os
import re
import shutil
import tempfile
from pathlib import Path

# Suppress tokenizers parallelism warning (must be before any HF imports)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
load_dotenv()

import mlx_whisper
import requests
from celery import Celery
from mlx_lm import generate, load
from pydub import AudioSegment

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

WHISPER_MODEL = "mlx-community/distil-whisper-large-v3"
LLM_MODEL = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"

app = Celery(
    "podcast_purifier_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_queues={
        "audio_processing": {"exchange": "audio_processing", "routing_key": "audio_processing"},
    },
    task_default_queue="audio_processing",
    # Only process one task at a time - MLX models don't support concurrent inference
    worker_concurrency=1,
    worker_prefetch_multiplier=1,
)

# Cache loaded models at module level
_llm_model = None
_llm_tokenizer = None


def get_llm():
    """Lazy-load and cache the LLM model."""
    global _llm_model, _llm_tokenizer
    if _llm_model is None:
        _llm_model, _llm_tokenizer = load(LLM_MODEL)
    return _llm_model, _llm_tokenizer


def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribe audio using mlx_whisper with word-level timestamps.

    Returns the full transcription result with word timestamps.
    """
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=WHISPER_MODEL,
        word_timestamps=True,
    )
    return result


def detect_ad_segments(transcript: dict) -> list[tuple[float, float]]:
    """
    Use LLM to analyze transcript and identify advertisement segments.

    Returns list of (start_ms, end_ms) tuples for ad segments.
    """
    model, tokenizer = get_llm()

    segments = transcript.get("segments", [])
    if not segments:
        return []

    # Build transcript text with timestamps for context
    transcript_lines = []
    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        text = seg["text"].strip()
        transcript_lines.append(f"[{start:.1f}s - {end:.1f}s]: {text}")

    transcript_text = "\n".join(transcript_lines)

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert at identifying advertisements in podcast transcripts. Your task is to find ALL segments that are ads, sponsorships, or promotional content. Be thorough - it's better to mark something as an ad than to miss one.

Look for these indicators:

SPONSORSHIP LANGUAGE:
- "This episode/podcast is sponsored by", "brought to you by", "thanks to our sponsor"
- "Today's sponsor", "this week's sponsor", "our partners at"
- "Speaking of [product category]", "And now a word from"

PROMO CODES AND OFFERS:
- "Use code", "promo code", "discount code", "coupon code"
- "percent off", "dollars off", "% off", "$ off"
- "Free trial", "free month", "free shipping", "money back guarantee"
- "Limited time offer", "exclusive offer", "special offer for listeners"

CALLS TO ACTION:
- "Go to", "Head to", "Visit", "Check out"
- "Link in the show notes/description", "click the link"
- "Sign up", "Get started", "Try it free", "Download the app"
- URLs mentioned (anything.com/something)

COMMON ADVERTISERS (these are almost always ads when mentioned):
- BetterHelp, Talkspace, Calm, Headspace (mental health)
- Squarespace, Shopify, Wix (websites)
- HelloFresh, Factor, Blue Apron (meal kits)
- NordVPN, ExpressVPN, Surfshark (VPNs)
- Athletic Greens, AG1, Liquid IV (supplements)
- SimpliSafe, Ring, ADT (security)
- ZipRecruiter, Indeed, LinkedIn (job sites)
- Audible, Kindle (Amazon)
- Stamps.com, ShipStation (shipping)
- Quip, Babbel, Duolingo

AD TRANSITIONS:
- "We'll be right back", "Quick break", "Let's take a break"
- "Back to the show", "Now back to", "Returning to"
- "Before we continue", "Before we get into"

HOST-READ ADS (the host talks about a product/service):
- Personal testimonials about products ("I've been using", "I love this product")
- Detailed product descriptions with specific benefits
- Mentioning getting a "special deal" or "worked out a deal"

PRE-ROLL ADS (at the very beginning, before the actual episode - BE AGGRESSIVE HERE):
- The first 0-180 seconds often contain ads BEFORE the actual show starts
- Content at the start that's unrelated to the episode topic
- Network/studio promos ("From Wondery", "A Spotify Original", "From the makers of", "iHeart", "Pushkin")
- Promos for other podcasts ("If you like this show, check out...", "New from...")
- Sponsor messages before any episode content
- Generic intros immediately followed by ads
- Dynamically inserted ads (often sound slightly different in audio quality)
- If the first segment mentions ANY product, service, or other podcast - it's probably an ad
- The actual episode usually starts with: the host greeting by name, episode topic introduction, or consistent theme music

Respond ONLY with a JSON array of ad segments. Each segment should have "start" and "end" times in seconds.
If there are no ads, respond with an empty array: []
Be aggressive - when in doubt, mark it as an ad.

Example response format:
[{{"start": 45.0, "end": 120.5}}, {{"start": 890.0, "end": 950.0}}]<|eot_id|><|start_header_id|>user<|end_header_id|>

Analyze this podcast transcript and identify ALL advertisement segments:

{transcript_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=1024,
        verbose=False,
    )

    # Parse the JSON response
    ad_segments = []
    try:
        # Extract JSON array from response - use greedy match to get the full array
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            segments_data = json.loads(json_match.group())
            for seg in segments_data:
                start_ms = float(seg["start"]) * 1000
                end_ms = float(seg["end"]) * 1000
                ad_segments.append((start_ms, end_ms))
        else:
            print(f"[AD_DETECTION] No JSON array found in LLM response: {response[:500]}")
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"[AD_DETECTION] Failed to parse LLM response: {e}")
        print(f"[AD_DETECTION] Raw response: {response[:500]}")

    return ad_segments


def merge_overlapping_segments(segments: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping or adjacent ad segments."""
    if not segments:
        return []

    sorted_segments = sorted(segments, key=lambda x: x[0])
    merged = [sorted_segments[0]]

    for start, end in sorted_segments[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 500:  # Allow 500ms gap for merging
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def remove_ad_segments(
    audio_path: str,
    ad_segments: list[tuple[float, float]],
    output_path: str,
    crossfade_ms: int = 500,
) -> None:
    """
    Remove ad segments from audio and apply crossfades at cut points.

    Args:
        audio_path: Path to input audio file
        ad_segments: List of (start_ms, end_ms) tuples to remove
        output_path: Path for output audio file
        crossfade_ms: Crossfade duration in milliseconds
    """
    audio = AudioSegment.from_mp3(audio_path)

    if not ad_segments:
        audio.export(output_path, format="mp3")
        return

    # Sort and merge overlapping segments
    ad_segments = merge_overlapping_segments(ad_segments)

    # Build list of segments to keep
    keep_segments = []
    current_pos = 0

    for ad_start, ad_end in ad_segments:
        if ad_start > current_pos:
            keep_segments.append((current_pos, ad_start))
        current_pos = ad_end

    # Add final segment after last ad
    if current_pos < len(audio):
        keep_segments.append((current_pos, len(audio)))

    if not keep_segments:
        # Edge case: entire audio is ads
        AudioSegment.empty().export(output_path, format="mp3")
        return

    # Extract and join segments with crossfade
    result = audio[keep_segments[0][0]:keep_segments[0][1]]

    for start, end in keep_segments[1:]:
        segment = audio[start:end]
        # Apply crossfade if both segments are long enough
        if len(result) >= crossfade_ms and len(segment) >= crossfade_ms:
            result = result.append(segment, crossfade=crossfade_ms)
        else:
            result = result + segment

    result.export(output_path, format="mp3")


def report_status(callback_url: str, episode_id: str, status: str, stage: str | None = None, error_message: str | None = None):
    """Report status change to the Manager."""
    # Derive base URL from callback_url (e.g., http://host/api/upload/1 -> http://host/api)
    # callback_url format: {base}/api/upload/{episode_id}
    base_url = callback_url.rsplit("/upload/", 1)[0]
    status_url = f"{base_url}/episodes/{episode_id}/status"
    try:
        payload = {"status": status, "stage": stage}
        if error_message:
            payload["error_message"] = error_message
        requests.post(status_url, json=payload, timeout=10)
    except Exception:
        pass  # Don't fail the task if status update fails


@app.task(bind=True, name="worker.process_episode")
def process_episode(self, episode_id: str, audio_url: str, callback_url: str) -> dict:
    """
    Process a podcast episode to strip ads.

    Pipeline:
    1. Download audio from audio_url
    2. Transcribe with mlx_whisper (word timestamps)
    3. Detect ad segments using LLM analysis
    4. Remove ad segments with crossfade
    5. Upload cleaned audio to callback_url

    Args:
        episode_id: Unique identifier for the episode
        audio_url: URL to download the source audio from
        callback_url: URL to upload the processed audio to
    """
    temp_dir = tempfile.mkdtemp(prefix="podcast_purifier_")
    input_path = Path(temp_dir) / f"{episode_id}_input.mp3"
    output_path = Path(temp_dir) / f"{episode_id}_cleaned.mp3"

    try:
        # Step 1: Download
        report_status(callback_url, episode_id, "processing", "downloading")
        self.update_state(
            state="DOWNLOADING",
            meta={"episode_id": episode_id, "step": "downloading audio"},
        )
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(audio_url, stream=True, timeout=600, headers=headers)
        response.raise_for_status()

        with open(input_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Step 2: Transcribe
        report_status(callback_url, episode_id, "processing", "transcribing")
        self.update_state(
            state="TRANSCRIBING",
            meta={"episode_id": episode_id, "step": "transcribing audio"},
        )
        transcript = transcribe_audio(str(input_path))

        # Step 3: Detect ads
        report_status(callback_url, episode_id, "processing", "analyzing")
        self.update_state(
            state="ANALYZING",
            meta={"episode_id": episode_id, "step": "detecting advertisements"},
        )
        ad_segments = detect_ad_segments(transcript)

        # Step 4: Remove ads
        report_status(callback_url, episode_id, "processing", "cutting")
        self.update_state(
            state="CUTTING",
            meta={
                "episode_id": episode_id,
                "step": "removing ad segments",
                "ad_segments_found": len(ad_segments),
            },
        )
        remove_ad_segments(str(input_path), ad_segments, str(output_path))

        # Step 5: Upload
        report_status(callback_url, episode_id, "processing", "uploading")
        self.update_state(
            state="UPLOADING",
            meta={"episode_id": episode_id, "step": "uploading cleaned audio"},
        )
        with open(output_path, "rb") as f:
            files = {"file": (f"{episode_id}.mp3", f, "audio/mpeg")}
            upload_response = requests.post(
                callback_url,
                files=files,
                data={"episode_id": episode_id},
                timeout=600,
            )
            upload_response.raise_for_status()

        return {
            "status": "success",
            "episode_id": episode_id,
            "ad_segments_removed": len(ad_segments),
            "message": "Episode processed and uploaded successfully",
        }

    except requests.RequestException as e:
        error_msg = f"Network error: {e}"
        report_status(callback_url, episode_id, "failed", None, error_msg)
        return {
            "status": "error",
            "episode_id": episode_id,
            "error_type": "network",
            "message": error_msg,
        }
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        report_status(callback_url, episode_id, "failed", None, error_msg)
        return {
            "status": "error",
            "episode_id": episode_id,
            "error_type": "processing",
            "message": error_msg,
        }

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    app.start()
