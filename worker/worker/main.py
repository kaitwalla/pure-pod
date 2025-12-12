import json
import logging
import os
import re
import shutil
import tempfile
import warnings
from pathlib import Path

# Suppress tokenizers parallelism warning (must be before any HF imports)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress torchaudio deprecation warnings about future torchcodec changes
warnings.filterwarnings("ignore", message=".*torchaudio.*torchcodec.*")

from dotenv import load_dotenv
load_dotenv()

import mlx_whisper
import requests
import torch

# PyTorch 2.6+ defaults weights_only=True for security, but pyannote models need these globals
# This must be done before importing pyannote.audio
torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])

# Additional pyannote-specific classes that need to be allowlisted for model loading
# PyTorch 2.6+ requires these to be explicitly marked as safe for deserialization
from pyannote.audio.core.task import Specifications, Problem, Resolution
from pyannote.audio.utils.powerset import Powerset
torch.serialization.add_safe_globals([Specifications, Problem, Resolution, Powerset])

from celery import Celery
from mlx_lm import generate, load
from pyannote.audio import Pipeline as DiarizationPipeline
from pydub import AudioSegment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
HF_TOKEN = os.environ.get("HF_TOKEN")  # Required for pyannote
WORKER_API_KEY = os.environ.get("WORKER_API_KEY")  # Required for upload authentication

WHISPER_MODEL = "mlx-community/distil-whisper-large-v3"
LLM_MODEL = "mlx-community/Qwen2.5-14B-Instruct-4bit"
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

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
_diarization_pipeline = None
_startup_validated = False


def validate_startup():
    """
    Validate all models and dependencies are accessible before processing tasks.
    Call this at worker startup to fail fast if something is misconfigured.
    """
    global _startup_validated
    if _startup_validated:
        return True

    logger.info("=" * 60)
    logger.info("[STARTUP] Validating worker configuration...")
    logger.info("=" * 60)

    errors = []

    # Check required environment variables
    if not HF_TOKEN:
        errors.append("HF_TOKEN environment variable not set (required for diarization)")
    if not WORKER_API_KEY:
        errors.append("WORKER_API_KEY environment variable not set (required for upload authentication)")

    # Validate diarization pipeline (this will fail if gated model access is denied)
    logger.info("[STARTUP] Loading diarization pipeline...")
    try:
        pipeline = get_diarization_pipeline()
        if pipeline is None:
            errors.append("Diarization pipeline failed to load (check HF_TOKEN)")
        else:
            logger.info("[STARTUP] ✓ Diarization pipeline loaded successfully")
    except Exception as e:
        errors.append(f"Diarization pipeline error: {e}")

    # Validate LLM
    logger.info("[STARTUP] Loading LLM model...")
    try:
        model, tokenizer = get_llm()
        logger.info("[STARTUP] ✓ LLM model loaded successfully")
    except Exception as e:
        errors.append(f"LLM model error: {e}")

    # Validate Whisper (just check import works, model downloads on first use)
    logger.info("[STARTUP] Checking Whisper availability...")
    try:
        # Do a minimal transcription test to ensure model downloads
        import tempfile
        import numpy as np
        from pydub import AudioSegment

        # Create a tiny silent audio file for testing
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            silent = AudioSegment.silent(duration=100)  # 100ms
            silent.export(f.name, format="wav")
            result = mlx_whisper.transcribe(f.name, path_or_hf_repo=WHISPER_MODEL)
        logger.info("[STARTUP] ✓ Whisper model loaded successfully")
    except Exception as e:
        errors.append(f"Whisper model error: {e}")

    if errors:
        logger.error("=" * 60)
        logger.error("[STARTUP] VALIDATION FAILED - Worker cannot start")
        logger.error("=" * 60)
        for error in errors:
            logger.error(f"  ✗ {error}")
        logger.error("=" * 60)
        raise RuntimeError(f"Startup validation failed: {'; '.join(errors)}")

    logger.info("=" * 60)
    logger.info("[STARTUP] ✓ All validations passed - Worker ready")
    logger.info("=" * 60)
    _startup_validated = True
    return True


@app.on_after_configure.connect
def setup_startup_validation(sender, **kwargs):
    """Run validation when Celery is configured."""
    validate_startup()


def get_diarization_pipeline():
    """Lazy-load and cache the diarization pipeline."""
    global _diarization_pipeline
    if _diarization_pipeline is None:
        if not HF_TOKEN:
            logger.warning("[DIARIZATION] HF_TOKEN not set, diarization will be skipped")
            return None
        logger.info("[DIARIZATION] Loading pyannote speaker diarization pipeline...")
        _diarization_pipeline = DiarizationPipeline.from_pretrained(
            DIARIZATION_MODEL,
            use_auth_token=HF_TOKEN
        )
        # Use MPS (Apple Silicon GPU) if available
        if torch.backends.mps.is_available():
            _diarization_pipeline.to(torch.device("mps"))
            logger.info("[DIARIZATION] Using MPS (Apple Silicon GPU)")
        else:
            logger.info("[DIARIZATION] Using CPU")
    return _diarization_pipeline


def get_llm():
    """Lazy-load and cache the LLM model."""
    global _llm_model, _llm_tokenizer
    if _llm_model is None:
        _llm_model, _llm_tokenizer = load(LLM_MODEL)
        # Add end-of-turn tokens to EOS token IDs so generation stops properly
        # Qwen uses <|im_end|>, Llama 3 uses <|eot_id|>
        for token in ["<|im_end|>", "<|eot_id|>"]:
            try:
                token_id = _llm_tokenizer.encode(token, add_special_tokens=False)[0]
                if hasattr(_llm_tokenizer, 'eos_token_ids'):
                    _llm_tokenizer.eos_token_ids.add(token_id)
                    logger.info(f"[LLM] Added {token} ({token_id}) to eos_token_ids")
            except Exception:
                pass  # Token doesn't exist in this model's vocabulary
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


def diarize_audio(audio_path: str) -> list[tuple[float, float, str]]:
    """
    Perform speaker diarization on audio.

    Returns list of (start_seconds, end_seconds, speaker_id) tuples.
    """
    pipeline = get_diarization_pipeline()
    if pipeline is None:
        return []

    # Convert MP3 to WAV to avoid torchaudio tensor size mismatch issues
    # See: https://github.com/pyannote/pyannote-audio/issues/1752
    wav_path = audio_path
    temp_wav = None
    if audio_path.lower().endswith('.mp3'):
        logger.info("[DIARIZATION] Converting MP3 to WAV for processing...")
        temp_wav = audio_path.rsplit('.', 1)[0] + '_diarize.wav'
        audio = AudioSegment.from_mp3(audio_path)
        audio.export(temp_wav, format="wav")
        wav_path = temp_wav
        logger.info(f"[DIARIZATION] Converted to WAV: {wav_path}")

    try:
        logger.info("[DIARIZATION] Running speaker diarization...")
        # Limit max speakers - most podcasts have 2-6 speakers (hosts + guests + ads)
        diarization = pipeline(wav_path, max_speakers=10)

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append((turn.start, turn.end, speaker))

        logger.info(f"[DIARIZATION] Found {len(segments)} speaker segments")

        # Log speaker summary
        speakers = set(s[2] for s in segments)
        for speaker in sorted(speakers):
            speaker_time = sum(s[1] - s[0] for s in segments if s[2] == speaker)
            logger.info(f"[DIARIZATION] {speaker}: {speaker_time:.1f}s total")

        return segments
    finally:
        # Clean up temporary WAV file
        if temp_wav and os.path.exists(temp_wav):
            os.remove(temp_wav)
            logger.info(f"[DIARIZATION] Cleaned up temp WAV file")


def get_speaker_at_time(diarization: list[tuple[float, float, str]], time_seconds: float) -> str:
    """Get the speaker ID at a given time, or 'UNKNOWN' if not found."""
    for start, end, speaker in diarization:
        if start <= time_seconds <= end:
            return speaker
    return "UNKNOWN"


def merge_transcript_with_diarization(
    transcript: dict,
    diarization: list[tuple[float, float, str]]
) -> dict:
    """
    Add speaker labels to transcript segments based on diarization.

    Returns transcript with speaker info added to each segment.
    """
    if not diarization:
        return transcript

    segments = transcript.get("segments", [])
    for seg in segments:
        # Use the midpoint of the segment to determine speaker
        midpoint = (seg["start"] + seg["end"]) / 2
        seg["speaker"] = get_speaker_at_time(diarization, midpoint)

    return transcript


def detect_ad_segments(
    transcript: dict,
    title: str | None = None,
    description: str | None = None,
) -> list[tuple[float, float]]:
    """
    Detect ad segments using a 3-pass approach:

    Pass 1: Find brand ads (URLs, promo codes, sponsor phrases)
    Pass 2: Find incongruous content (doesn't match episode description or surrounding content)
    Pass 3: Verify and refine start/end boundaries

    Args:
        transcript: Whisper transcript with segments
        title: Episode title from RSS feed
        description: Episode description/show notes from RSS feed

    Returns:
        List of (start_ms, end_ms) tuples for ad segments.
    """
    model, tokenizer = get_llm()

    segments = transcript.get("segments", [])
    if not segments:
        return []

    # Build episode context from metadata
    episode_context = _build_episode_context(title, description)

    # === PASS 1: Find brand ads (URLs, promo codes, sponsor phrases) ===
    pass1_ads = _pass1_find_brand_ads(segments, episode_context, model, tokenizer)

    # === PASS 2: Find incongruous content ===
    pass2_ads = _pass2_find_incongruous_content(segments, episode_context, pass1_ads, model, tokenizer)

    # Combine pass 1 and pass 2 results
    all_ads = pass1_ads + pass2_ads
    if all_ads:
        all_ads = merge_overlapping_segments(all_ads)

    # === PASS 3: Verify and refine boundaries ===
    if all_ads:
        all_ads = _pass3_verify_boundaries(segments, episode_context, all_ads, model, tokenizer)

    return all_ads


def _build_episode_context(title: str | None, description: str | None) -> str:
    """Build episode context string from title and description."""
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if description:
        # Strip HTML tags and truncate description
        import re
        clean_desc = re.sub(r'<[^>]+>', '', description)
        clean_desc = re.sub(r'\s+', ' ', clean_desc).strip()
        if len(clean_desc) > 2000:
            clean_desc = clean_desc[:2000] + "..."
        parts.append(f"Description: {clean_desc}")

    if not parts:
        return "No episode metadata available."

    return "\n".join(parts)


def _pass1_find_brand_ads(
    segments: list,
    episode_context: str,
    model,
    tokenizer
) -> list[tuple[float, float]]:
    """
    Pass 1: Find brand ads using deterministic pattern matching + LLM boundary refinement.

    Looks for: URLs, promo codes, sponsor phrases, brand mentions with promotional context.
    """
    # Find segments with ad indicators (deterministic)
    ad_indicator_segments = _find_ad_indicator_segments(segments)

    if not ad_indicator_segments:
        return []

    # Group nearby indicators (within 2 minutes) as likely same ad block
    indicator_groups = []
    current_group = [ad_indicator_segments[0]]

    for indicator in ad_indicator_segments[1:]:
        if indicator["start"] - current_group[-1]["end"] < 120:
            current_group.append(indicator)
        else:
            indicator_groups.append(current_group)
            current_group = [indicator]
    indicator_groups.append(current_group)

    # For each group, find full ad boundaries using LLM
    ad_segments = []
    for group in indicator_groups:
        group_start = group[0]["start"]
        group_end = group[-1]["end"]

        # Get context: 90 seconds before, 30 seconds after
        context_start = max(0, group_start - 90)
        context_end = min(segments[-1]["end"], group_end + 30)

        # Build context transcript
        context_lines = []
        for seg in segments:
            if seg["start"] >= context_start and seg["end"] <= context_end:
                speaker = seg.get("speaker", "")
                prefix = f"{speaker}: " if speaker else ""
                context_lines.append(f"[{seg['start']:.1f}s] {prefix}{seg['text'].strip()}")

        context_text = "\n".join(context_lines)

        # What we found in this group
        indicators_found = ", ".join([f"'{i['matched']}' at {i['start']:.0f}s" for i in group])

        prompt = f"""<|im_start|>system
You are finding the exact boundaries of an advertisement in a podcast.

EPISODE INFO:
{episode_context}

AD INDICATORS FOUND (these are definitely part of an ad):
{indicators_found}

Find where this ad STARTS and ENDS.

AD START - look backwards from the first indicator for:
- "brought to you by", "sponsored by", "speaking of which", "a]word from our sponsor"
- Sudden topic change from episode content to product/service promotion
- Speaker change to someone reading an ad

AD END - look forwards from the last indicator for:
- Return to episode topic (see episode info above)
- "anyway", "so", "back to", "alright" followed by episode-related content
- Speaker change back to main host discussing episode content

Return ONLY: {{"start":SECONDS,"end":SECONDS}}<|im_end|>
<|im_start|>user
{context_text}<|im_end|>
<|im_start|>assistant
"""

        response = generate(model, tokenizer, prompt=prompt, max_tokens=100, verbose=False)

        # Parse response
        for end_token in ["<|im_end|>", "<|eot_id|>"]:
            if end_token in response:
                response = response.split(end_token)[0]
                break

        try:
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                boundary = json.loads(json_match.group())
                start_ms = float(boundary["start"]) * 1000
                end_ms = float(boundary["end"]) * 1000
                if end_ms - start_ms >= 10000:  # Min 10 seconds
                    ad_segments.append((start_ms, end_ms))
        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback: use indicator group boundaries with padding
            start_ms = max(0, (group_start - 30)) * 1000
            end_ms = (group_end + 10) * 1000
            ad_segments.append((start_ms, end_ms))

    return ad_segments


def _pass2_find_incongruous_content(
    segments: list,
    episode_context: str,
    existing_ads: list[tuple[float, float]],
    model,
    tokenizer
) -> list[tuple[float, float]]:
    """
    Pass 2: Find content that is incongruous with both:
    - The episode description (off-topic)
    - The surrounding content (contextually out of place)

    This catches brand ads without URLs (e.g., "Sprite", "McDonald's").
    """
    # Build transcript with existing ads marked
    def is_in_existing_ad(start_s, end_s):
        start_ms = start_s * 1000
        end_ms = end_s * 1000
        for ad_start, ad_end in existing_ads:
            if start_ms < ad_end and end_ms > ad_start:
                return True
        return False

    transcript_lines = []
    for seg in segments:
        speaker = seg.get("speaker", "")
        prefix = f"{speaker}: " if speaker else ""
        marker = "[ALREADY MARKED AS AD] " if is_in_existing_ad(seg["start"], seg["end"]) else ""
        transcript_lines.append(f"{marker}[{seg['start']:.1f}s] {prefix}{seg['text'].strip()}")

    transcript_text = "\n".join(transcript_lines)

    # Truncate if needed
    if len(transcript_text) > 120000:
        transcript_text = transcript_text[:120000] + "\n[truncated]"

    prompt = f"""<|im_start|>system
You are finding advertisements that were MISSED in a first pass.

EPISODE INFO (this is what the episode is actually about):
{episode_context}

Some segments are already marked as [ALREADY MARKED AS AD] - ignore those.

Look for UNMARKED segments that are:
1. OFF-TOPIC: Not related to the episode topic above
2. PROMOTIONAL: Promoting a product, service, or brand
3. INCONGRUOUS: Don't fit with what comes before/after

Examples of what to find:
- Brand mentions with promotional language ("refreshing Sprite", "delicious McDonald's")
- Product pitches without URLs ("try our new service", "you'll love this app")
- Sponsor messages that slipped through

Examples of what is NOT an ad:
- Discussion related to the episode topic (even if mentioning brands as examples)
- The podcast promoting its own social media or Patreon
- Casual conversation between hosts

Return JSON array of NEW ad segments only: [{{"start":SECONDS,"end":SECONDS}}]
Return [] if no additional ads found.<|im_end|>
<|im_start|>user
{transcript_text}<|im_end|>
<|im_start|>assistant
"""

    response = generate(model, tokenizer, prompt=prompt, max_tokens=1024, verbose=False)

    # Parse response
    for end_token in ["<|im_end|>", "<|eot_id|>"]:
        if end_token in response:
            response = response.split(end_token)[0]
            break

    new_ads = []
    try:
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        response = re.sub(r'(\d+\.?\d*)s([,\}\]])', r'\1\2', response)

        json_match = re.search(r'\[\s*\{[\s\S]*?\}\s*\]', response)
        if json_match:
            segments_data = json.loads(json_match.group())
            for seg in segments_data:
                start_ms = float(seg["start"]) * 1000
                end_ms = float(seg["end"]) * 1000
                if end_ms - start_ms >= 10000:  # Min 10 seconds
                    new_ads.append((start_ms, end_ms))
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    return new_ads


def _pass3_verify_boundaries(
    segments: list,
    episode_context: str,
    ad_segments: list[tuple[float, float]],
    model,
    tokenizer
) -> list[tuple[float, float]]:
    """
    Pass 3: Verify and refine ad boundaries.

    Reviews the marked transcript and adjusts boundaries where:
    - Ad started too late (missed the intro)
    - Ad ended too early (cut off mid-ad)
    - Ad ended too late (included episode content)
    """
    # Build marked transcript
    def is_in_ad(start_s, end_s):
        start_ms = start_s * 1000
        end_ms = end_s * 1000
        for ad_start, ad_end in ad_segments:
            if start_ms < ad_end and end_ms > ad_start:
                return True
        return False

    marked_lines = []
    for seg in segments:
        speaker = seg.get("speaker", "")
        prefix = f"{speaker}: " if speaker else ""
        marker = "[REMOVE]" if is_in_ad(seg["start"], seg["end"]) else "[KEEP]"
        marked_lines.append(f"{marker} [{seg['start']:.1f}s] {prefix}{seg['text'].strip()}")

    marked_transcript = "\n".join(marked_lines)

    # Truncate if needed
    if len(marked_transcript) > 120000:
        marked_transcript = marked_transcript[:120000] + "\n[truncated]"

    prompt = f"""<|im_start|>system
You are verifying ad boundaries in a podcast transcript.

EPISODE INFO:
{episode_context}

[REMOVE] = will be cut from audio
[KEEP] = will remain in final audio

Check for these boundary errors:

1. STARTED TOO LATE - Look at [KEEP] lines RIGHT BEFORE [REMOVE] blocks:
   - Are they ad intros? ("speaking of which", "let me tell you about", "brought to you by")
   - If yes, include them in the ad

2. ENDED TOO EARLY - Look at [KEEP] lines RIGHT AFTER [REMOVE] blocks:
   - Are they still ad content? (still promoting the product, final call-to-action)
   - If yes, extend the ad

3. ENDED TOO LATE - Look at the END of [REMOVE] blocks:
   - Did we include episode content? (host returns to topic, asks guest a question)
   - If yes, the ad should have ended earlier

Return corrections as JSON:
- "adjust": segments where boundaries need fixing - [{{"original_start":SEC,"original_end":SEC,"new_start":SEC,"new_end":SEC}}]

Return {{"adjust":[]}} if boundaries are correct.<|im_end|>
<|im_start|>user
{marked_transcript}<|im_end|>
<|im_start|>assistant
"""

    response = generate(model, tokenizer, prompt=prompt, max_tokens=1024, verbose=False)

    # Parse response
    for end_token in ["<|im_end|>", "<|eot_id|>"]:
        if end_token in response:
            response = response.split(end_token)[0]
            break

    response = re.sub(r'```json\s*', '', response)
    response = re.sub(r'```\s*', '', response)
    response = re.sub(r'(\d+\.?\d*)s([,\}\]])', r'\1\2', response)

    try:
        json_match = re.search(r'\{[\s\S]*"adjust"[\s\S]*\}', response)
        if json_match:
            corrections = json.loads(json_match.group())
            adjustments = corrections.get("adjust", [])

            if adjustments:
                # Apply adjustments
                adjusted_ads = list(ad_segments)
                for adj in adjustments:
                    orig_start_ms = float(adj.get("original_start", 0)) * 1000
                    orig_end_ms = float(adj.get("original_end", 0)) * 1000
                    new_start_ms = float(adj.get("new_start", orig_start_ms / 1000)) * 1000
                    new_end_ms = float(adj.get("new_end", orig_end_ms / 1000)) * 1000

                    # Find and replace the matching segment
                    for i, (start, end) in enumerate(adjusted_ads):
                        # Match if close enough (within 5 seconds)
                        if abs(start - orig_start_ms) < 5000 and abs(end - orig_end_ms) < 5000:
                            if new_end_ms - new_start_ms >= 10000:  # Min 10 seconds
                                adjusted_ads[i] = (new_start_ms, new_end_ms)
                            break

                # Re-merge in case adjustments created overlaps
                return merge_overlapping_segments(adjusted_ads)
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    return ad_segments


def _find_ad_indicator_segments(segments: list) -> list[dict]:
    """
    Deterministic scan for segments containing URLs or promo codes.
    Returns list of dicts with segment info and matched pattern.
    """
    # URL patterns - looking for promotional URLs and ad indicators
    url_patterns = [
        # URLs
        r'\b\w+\.com\b',  # something.com
        r'\b\w+\.org\b',  # something.org
        r'\b\w+\.co\b',   # something.co
        r'\b\w+\s+dot\s+com\b',  # "something dot com"
        r'\bslash\s+\w+',  # "slash podcastname" (spoken URL)
        r'\bforward\s+slash',  # "forward slash"
        # Call to action phrases
        r'\bvisit\s+\w+',  # "visit sitename"
        r'\bgo\s+to\s+\w+',  # "go to sitename"
        r'\bhead\s+to\s+\w+',  # "head to sitename"
        r'\bhead\s+over\s+to',  # "head over to"
        r'\bcheck\s+out\s+\w+',  # "check out sitename"
        r'\bsign\s+up\s+(at|for)',  # "sign up at/for"
        r'\bdownload\s+(the\s+)?app',  # "download the app"
        # Promo codes
        r'\bpromo\s*code\b',  # promo code
        r'\bcode\s+[A-Z0-9]+\b',  # code SAVE20
        r'\buse\s+code\b',  # use code
        r'\bdiscount\s+code\b',  # discount code
        r'\benter\s+code\b',  # enter code
        # Offers
        r'\b\d+%\s*off\b',  # 50% off
        r'\bfree\s+trial\b',  # free trial
        r'\bfirst\s+\w+\s+free\b',  # first month free
        r'\bfree\s+shipping\b',  # free shipping
        r'\bmoney[\s-]back\s+guarantee\b',  # money-back guarantee
        # Sponsor phrases (strong ad indicators)
        r'\bbrought\s+to\s+you\s+by\b',  # "brought to you by"
        r'\bsponsored\s+by\b',  # "sponsored by"
        r'\bsupport(ed)?\s+(for\s+)?(this\s+)?(show|podcast|episode)\s+(comes?\s+from|is\s+brought)',  # "support for this show comes from"
        r'\btoday\'?s\s+sponsor\b',  # "today's sponsor"
        r'\bour\s+sponsor\b',  # "our sponsor"
        r'\bthis\s+(episode|show|podcast)\s+is\s+(brought|sponsored)',  # "this episode is brought/sponsored"
        r'\blet\s+me\s+tell\s+you\s+about\b',  # "let me tell you about" (host-read intro)
        r'\bword\s+from\s+(our\s+)?sponsor',  # "word from our sponsor"
    ]

    # Combine patterns
    combined_pattern = '|'.join(url_patterns)

    url_segments = []
    for seg in segments:
        text = seg["text"]
        match = re.search(combined_pattern, text, re.IGNORECASE)
        if match:
            url_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "start_ms": seg["start"] * 1000,
                "end_ms": seg["end"] * 1000,
                "text": text.strip(),
                "matched": match.group(),
                "speaker": seg.get("speaker", "")
            })

    return url_segments


def merge_overlapping_segments(segments: list[tuple[float, float]], gap_ms: float = 1000) -> list[tuple[float, float]]:
    """Merge overlapping or adjacent ad segments.

    Args:
        segments: List of (start_ms, end_ms) tuples
        gap_ms: Maximum gap between segments to merge (default 1 second)
    """
    if not segments:
        return []

    sorted_segments = sorted(segments, key=lambda x: x[0])
    merged = [sorted_segments[0]]

    for start, end in sorted_segments[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + gap_ms:
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
def process_episode(
    self,
    episode_id: str,
    audio_url: str,
    callback_url: str,
    title: str | None = None,
    description: str | None = None,
) -> dict:
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
        title: Episode title (for ad detection context)
        description: Episode description/show notes (for ad detection context)
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

        # Step 2b: Speaker diarization
        report_status(callback_url, episode_id, "processing", "diarizing")
        self.update_state(
            state="DIARIZING",
            meta={"episode_id": episode_id, "step": "identifying speakers"},
        )
        diarization = diarize_audio(str(input_path))

        # Merge speaker info into transcript
        if diarization:
            transcript = merge_transcript_with_diarization(transcript, diarization)

        # Step 3: Detect ads
        report_status(callback_url, episode_id, "processing", "analyzing")
        self.update_state(
            state="ANALYZING",
            meta={"episode_id": episode_id, "step": "detecting advertisements"},
        )
        ad_segments = detect_ad_segments(transcript, title=title, description=description)

        # Save cleaned transcript to desktop
        desktop_path = Path.home() / "Desktop"
        cleaned_transcript_path = desktop_path / f"episode_{episode_id}_cleaned.txt"

        def is_in_ad(start_s, end_s, ad_segs):
            """Check if a segment overlaps with any ad segment."""
            start_ms = start_s * 1000
            end_ms = end_s * 1000
            for ad_start, ad_end in ad_segs:
                if start_ms < ad_end and end_ms > ad_start:
                    return True
            return False

        with open(cleaned_transcript_path, "w") as f:
            f.write(f"=== CLEANED TRANSCRIPT (Episode {episode_id}) ===\n")
            f.write(f"=== Ad segments removed: {len(ad_segments)} ===\n\n")
            for i, (start_ms, end_ms) in enumerate(ad_segments):
                f.write(f"  [AD {i+1}] {start_ms/1000:.1f}s - {end_ms/1000:.1f}s (duration: {(end_ms-start_ms)/1000:.1f}s)\n")
            f.write("\n--- Transcript ---\n\n")
            for seg in transcript.get("segments", []):
                speaker = seg.get("speaker", "")
                prefix = f"{speaker}: " if speaker else ""
                if is_in_ad(seg["start"], seg["end"], ad_segments):
                    f.write(f"[REMOVED] [{seg['start']:.1f}s - {seg['end']:.1f}s] {prefix}{seg['text'].strip()}\n")
                else:
                    f.write(f"[KEPT]    [{seg['start']:.1f}s - {seg['end']:.1f}s] {prefix}{seg['text'].strip()}\n")

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
            upload_headers = {}
            if WORKER_API_KEY:
                upload_headers["X-API-Key"] = WORKER_API_KEY
            upload_response = requests.post(
                callback_url,
                files=files,
                data={"episode_id": episode_id},
                headers=upload_headers,
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
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    app.start()
