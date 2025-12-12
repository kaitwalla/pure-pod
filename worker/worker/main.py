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


def detect_ad_segments(transcript: dict) -> list[tuple[float, float]]:
    """
    Detect ad segments using a content-first approach:
    1. Find all URLs/promo codes (deterministic)
    2. Summarize the episode content (LLM)
    3. Use summary + URLs to find ad boundaries (LLM)

    Returns list of (start_ms, end_ms) tuples for ad segments.
    """
    model, tokenizer = get_llm()

    segments = transcript.get("segments", [])
    if not segments:
        return []

    # === STEP 1: Find all URLs and promo codes (deterministic) ===
    url_segments = _find_url_segments(segments)

    # === STEP 2: Summarize episode content (LLM) ===
    episode_summary = _summarize_episode_content(segments, model, tokenizer)

    # === STEP 3: Scan for ALL ads using summary (catches brand ads without URLs) ===
    ad_segments = _find_all_ads_with_summary(segments, episode_summary, model, tokenizer)

    # === STEP 4: For each URL group, refine boundaries (LLM) ===
    if not url_segments:
        if ad_segments:
            ad_segments = merge_overlapping_segments(ad_segments)
        return ad_segments

    # Group nearby URLs (within 2 minutes) as they're likely the same ad block
    url_groups = []
    current_group = [url_segments[0]]

    for url_seg in url_segments[1:]:
        # If this URL is within 2 minutes of the last one in current group, add to group
        if url_seg["start"] - current_group[-1]["end"] < 120:
            current_group.append(url_seg)
        else:
            url_groups.append(current_group)
            current_group = [url_seg]
    url_groups.append(current_group)

    # For each URL group, find the full ad boundaries and add to our list
    url_based_ads = []
    for group_idx, url_group in enumerate(url_groups):
        group_start = url_group[0]["start"]
        group_end = url_group[-1]["end"]

        # Get context: 90 seconds before first URL, 30 seconds after last URL
        context_start = max(0, group_start - 90)
        context_end = min(segments[-1]["end"], group_end + 30)

        # Build context transcript
        context_lines = []
        for seg in segments:
            if seg["start"] >= context_start and seg["end"] <= context_end:
                speaker = seg.get("speaker", "")
                if speaker:
                    context_lines.append(f"[{seg['start']:.1f}s] {speaker}: {seg['text'].strip()}")
                else:
                    context_lines.append(f"[{seg['start']:.1f}s]: {seg['text'].strip()}")

        context_text = "\n".join(context_lines)

        # URLs in this group
        urls_found = ", ".join([f"'{u['matched']}' at {u['start']:.0f}s" for u in url_group])

        prompt = f"""<|im_start|>system
You are finding the exact boundaries of an advertisement in a podcast.

EPISODE TOPIC (this is what the real content is about):
{episode_summary}

URLs/PROMO CODES FOUND (these are definitely part of ads):
{urls_found}

Your task: Find where this ad STARTS and ENDS.

AD START: Look backwards from the first URL for where they transition INTO the ad:
- "brought to you by", "sponsored by", "speaking of which"
- Topic suddenly changes from episode content to product promotion
- Different speaker starts talking about a product

AD END: Look forwards from the last URL for where they transition OUT of the ad:
- Host returns to the EPISODE TOPIC (see above)
- Speaker changes back to main host discussing episode content
- "anyway", "so", "back to" followed by episode-related content

IMPORTANT:
- The ad ends when they return to the EPISODE TOPIC, not when they stop mentioning the product
- Be conservative - don't include episode content in the ad boundaries

Return ONLY: {{"start":SECONDS,"end":SECONDS}}<|im_end|>
<|im_start|>user
{context_text}<|im_end|>
<|im_start|>assistant
"""

        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=100,
            verbose=False,
        )

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

                # Sanity check: ad should be at least 10 seconds
                if end_ms - start_ms >= 10000:
                    url_based_ads.append((start_ms, end_ms))
        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback: use URL group boundaries with some padding
            start_ms = max(0, (group_start - 30)) * 1000
            end_ms = (group_end + 10) * 1000
            url_based_ads.append((start_ms, end_ms))

    # Combine summary-based and URL-based ads
    all_ads = ad_segments + url_based_ads

    # Merge overlapping segments
    if all_ads:
        all_ads = merge_overlapping_segments(all_ads)

    return all_ads


def _find_all_ads_with_summary(
    segments: list,
    episode_summary: str,
    model,
    tokenizer
) -> list[tuple[float, float]]:
    """
    Use the episode summary to find ALL ads, including brand ads without URLs.
    """
    # Build transcript with timestamps
    transcript_lines = []
    for seg in segments:
        speaker = seg.get("speaker", "")
        if speaker:
            transcript_lines.append(f"[{seg['start']:.1f}s] {speaker}: {seg['text'].strip()}")
        else:
            transcript_lines.append(f"[{seg['start']:.1f}s]: {seg['text'].strip()}")

    transcript_text = "\n".join(transcript_lines)

    # Truncate if needed
    if len(transcript_text) > 150000:
        transcript_text = transcript_text[:150000] + "\n[truncated]"

    prompt = f"""<|im_start|>system
You are finding advertisements in a podcast transcript.

EPISODE CONTENT (what this episode is actually about):
{episode_summary}

YOUR TASK: Find all advertisements - segments that are NOT about the episode topic above.

Ads include:
- Brand promotions (e.g., "Sprite", "McDonald's") even without URLs
- Product/service pitches with URLs or promo codes
- Sponsor messages ("brought to you by", "sponsored by")
- Pre-roll and mid-roll ad breaks

NOT ads:
- Discussion of the episode's main topic (see summary above)
- The podcast promoting itself
- Casual host conversation

For each ad block, return start and end timestamps. Combine consecutive ads into one block.

Return JSON array: [{{"start":SECONDS,"end":SECONDS}}]
Return [] if no ads found.<|im_end|>
<|im_start|>user
{transcript_text}<|im_end|>
<|im_start|>assistant
"""

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=1024,
        verbose=False,
    )

    # Parse response
    for end_token in ["<|im_end|>", "<|eot_id|>"]:
        if end_token in response:
            response = response.split(end_token)[0]
            break

    ad_segments = []
    try:
        # Clean up response
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
                    ad_segments.append((start_ms, end_ms))
    except (json.JSONDecodeError, KeyError, TypeError):
        pass  # No valid response

    return ad_segments


def _find_url_segments(segments: list) -> list[dict]:
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


def _summarize_episode_content(segments: list, model, tokenizer) -> str:
    """
    Ask LLM to summarize the actual episode content, ignoring ads.
    This summary becomes the source of truth for what the episode is about.
    """
    # Build transcript text
    transcript_lines = []
    for seg in segments:
        start = seg["start"]
        text = seg["text"].strip()
        speaker = seg.get("speaker", "")
        if speaker:
            transcript_lines.append(f"[{start:.1f}s] {speaker}: {text}")
        else:
            transcript_lines.append(f"[{start:.1f}s]: {text}")

    transcript_text = "\n".join(transcript_lines)

    # Truncate if too long (we just need enough to understand the topic)
    if len(transcript_text) > 100000:
        transcript_text = transcript_text[:100000] + "\n[truncated]"

    prompt = f"""<|im_start|>system
You are summarizing a podcast episode. Write a 2-3 sentence summary of what this episode is ACTUALLY ABOUT - the main topic, discussion, story, or interview.

IMPORTANT: Ignore all advertisements, sponsors, and product promotions. Only summarize the real episode content.

Common ad indicators to IGNORE:
- Product promotions with URLs or promo codes
- "Brought to you by", "sponsored by" segments
- Pitches for services like VPNs, meal kits, mattresses, etc.

Focus ONLY on: What is the actual episode discussing? What's the main topic or story?<|im_end|>
<|im_start|>user
{transcript_text}<|im_end|>
<|im_start|>assistant
This episode is about"""

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=256,
        verbose=False,
    )

    # Clean up response
    for end_token in ["<|im_end|>", "<|eot_id|>"]:
        if end_token in response:
            response = response.split(end_token)[0]
            break

    summary = "This episode is about" + response.strip()
    return summary


def _refine_ad_segments(
    segments: list,
    ad_segments_ms: list[tuple[float, float]],
    model,
    tokenizer
) -> list[tuple[float, float]]:
    """
    Second pass: Review the marked transcript and refine ad boundaries.

    Shows the LLM which parts are marked as [KEEP] vs [REMOVE] and asks it to:
    1. Find any missed ads in [KEEP] sections
    2. Identify any real content incorrectly marked as [REMOVE]
    """
    # Deterministic check: find all segments with URLs/promo codes
    url_segments = _find_url_segments(segments)

    # Find URL segments that are NOT covered by current ad segments
    def is_covered(url_start, url_end):
        for ad_start, ad_end in ad_segments_ms:
            if url_start >= ad_start and url_end <= ad_end:
                return True
        return False

    uncovered_urls = [(s, e) for s, e in url_segments if not is_covered(s, e)]

    # Find and auto-fill short gaps between ad blocks (likely missed stacked ads)
    sorted_ads = sorted(ad_segments_ms, key=lambda x: x[0])
    auto_filled_gaps = []  # < 60s gaps - auto-fill these
    suspicious_gaps = []  # 60-180s gaps - ask LLM to check

    for i in range(len(sorted_ads) - 1):
        _, end_ms = sorted_ads[i]
        next_start_ms, _ = sorted_ads[i + 1]
        gap_seconds = (next_start_ms - end_ms) / 1000
        if 5 < gap_seconds < 60:
            # Auto-fill short gaps - these are almost certainly missed stacked ads
            auto_filled_gaps.append((end_ms, next_start_ms))
        elif 60 <= gap_seconds < 180:
            suspicious_gaps.append((end_ms / 1000, next_start_ms / 1000))

    # Add auto-filled gaps to ad segments and re-merge
    if auto_filled_gaps:
        ad_segments_ms = list(ad_segments_ms) + auto_filled_gaps
        ad_segments_ms = merge_overlapping_segments(ad_segments_ms)

    # Now build marked transcript with updated ad segments (including auto-filled gaps)
    def is_in_ad(start_s: float, end_s: float) -> bool:
        """Check if a segment overlaps with any ad segment."""
        start_ms = start_s * 1000
        end_ms = end_s * 1000
        for ad_start, ad_end in ad_segments_ms:
            if start_ms < ad_end and end_ms > ad_start:
                return True
        return False

    def has_url(start_s: float, end_s: float) -> bool:
        """Check if segment contains a URL (from our deterministic scan)."""
        start_ms = start_s * 1000
        end_ms = end_s * 1000
        for url_start, url_end in url_segments:
            if abs(start_ms - url_start) < 100 and abs(end_ms - url_end) < 100:
                return True
        return False

    marked_lines = []
    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        text = seg["text"].strip()
        speaker = seg.get("speaker", "")
        in_ad = is_in_ad(start, end)
        has_url_marker = has_url(start, end) and not in_ad
        if in_ad:
            marker = "[REMOVE]"
        elif has_url_marker:
            marker = "[KEEP][HAS URL!]"  # Flag URLs not in ad blocks
        else:
            marker = "[KEEP]"
        # Include speaker label if available
        if speaker:
            marked_lines.append(f"{marker} [{start:.1f}s - {end:.1f}s] {speaker}: {text}")
        else:
            marked_lines.append(f"{marker} [{start:.1f}s - {end:.1f}s]: {text}")

    marked_transcript = "\n".join(marked_lines)

    # Build warnings for prompt
    extra_warnings = ""

    # Warning about suspicious gaps
    if suspicious_gaps:
        gaps_list = ", ".join([f"{s:.0f}s-{e:.0f}s" for s, e in suspicious_gaps])
        extra_warnings += f"""
SUSPICIOUS GAPS BETWEEN AD BLOCKS (1-3 minutes):
{gaps_list}
Check if these [KEEP] sections contain product promotions - they may be missed ads.
"""

    # Warning about uncovered URLs - these are VERY likely missed ads
    if uncovered_urls:
        extra_warnings += """
CRITICAL - URLS FOUND IN [KEEP] SECTIONS:
Lines marked [KEEP][HAS URL!] contain URLs or promo codes but are NOT marked as ads.
These are almost certainly missed ads! Find the full ad boundaries around these URLs.
"""

    prompt = f"""<|im_start|>system
You are reviewing a podcast transcript marked for ad removal. Your job is to find errors.

[REMOVE] = will be cut from audio
[KEEP] = will remain in final audio

CHECK FOR THESE ERRORS:

1. BOUNDARY ERRORS - Did we start/end ads at the wrong place?
   - STARTED TOO LATE? Look at [KEEP] lines RIGHT BEFORE [REMOVE] blocks - are they the ad intro? (e.g., "Speaking of...", "You know what helps?", "brought to you by")
   - ENDED TOO EARLY? Look at [KEEP] lines RIGHT AFTER [REMOVE] blocks - still ad content? (e.g., final URL mention, "check them out")
   - ENDED TOO LATE? This is critical! Look at the END of each [REMOVE] block for signs the ad already ended:
     * CONTENT CHANGE: Host returns to episode topic, asks guest a question, references earlier discussion, continues the story
     * SPEAKER CHANGE: Main host(s) resume speaking after a different speaker did the ad read
     * TRANSITION PHRASES: "Anyway...", "So...", "Back to...", "As I was saying..."
     If you see these signs INSIDE a [REMOVE] block, the ad ended too early - return the correct end time in "remove"

2. MISSED ADS - Look through ALL [KEEP] sections for:
   - Product promotions with URLs (e.g., "visit example.com/podcast")
   - Promo codes (e.g., "use code SAVE20")
   - Sponsor mentions with calls-to-action
   - Host-read ads that blend in naturally

3. FALSE POSITIVES - [REMOVE] sections that are actually episode content (rare, but check)

4. SPEAKER CHANGES - If transcript has speaker labels (SPEAKER_00, etc.):
   - Different/new speakers often indicate pre-recorded ads
   - Same speaker but sudden topic change to product promotion = host-read ad
{extra_warnings}
Return corrections as JSON:
- "add": NEW segments to mark as ads (missed ads) - [{{"start":SECONDS,"end":SECONDS}}]
- "remove": segments currently marked [REMOVE] that should be KEPT (false positives / ad ended too late) - [{{"start":SECONDS,"end":SECONDS}}]

Example: If ad block 100s-200s actually ended at 150s, return {{"add":[],"remove":[{{"start":150,"end":200}}]}} to keep the content from 150-200s.

Return {{"add":[],"remove":[]}} if no corrections needed.
<|im_end|>
<|im_start|>user
{marked_transcript}<|im_end|>
<|im_start|>assistant
"""

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=1024,
        verbose=False,
    )

    # Parse corrections
    refined_ads = list(ad_segments_ms)  # Start with initial ads

    # Clean up response
    for end_token in ["<|im_end|>", "<|eot_id|>"]:
        if end_token in response:
            response = response.split(end_token)[0]
            break
    response = re.sub(r'```json\s*', '', response)
    response = re.sub(r'```\s*', '', response)
    response = re.sub(r'(\d+\.?\d*)s([,\}\]])', r'\1\2', response)

    try:
        # Look for JSON object with add/remove keys
        json_match = re.search(r'\{[\s\S]*"add"[\s\S]*"remove"[\s\S]*\}', response)
        if not json_match:
            json_match = re.search(r'\{[\s\S]*"remove"[\s\S]*"add"[\s\S]*\}', response)

        if json_match:
            json_str = json_match.group()
            corrections = json.loads(json_str)

            # Add missed ads
            for seg in corrections.get("add", []):
                start_ms = float(seg["start"]) * 1000
                end_ms = float(seg["end"]) * 1000
                if end_ms - start_ms >= 10000:  # Min 10 seconds
                    refined_ads.append((start_ms, end_ms))

            # Remove false positives
            for seg in corrections.get("remove", []):
                start_ms = float(seg["start"]) * 1000
                end_ms = float(seg["end"]) * 1000
                # Remove any ad that significantly overlaps with this segment
                refined_ads = [
                    (a_start, a_end) for a_start, a_end in refined_ads
                    if not (a_start < end_ms and a_end > start_ms)
                ]

    except (json.JSONDecodeError, KeyError, TypeError):
        pass  # Keep original ads if parsing fails

    # Merge overlapping segments
    if refined_ads:
        refined_ads = merge_overlapping_segments(refined_ads)

    return refined_ads


def _detect_ads_chunked(segments: list, model, tokenizer) -> list[tuple[float, float]]:
    """Process transcript in chunks for long episodes."""
    all_ad_segments = []

    # Process in chunks of ~100 segments (roughly 5-10 minutes each)
    CHUNK_SIZE = 100
    OVERLAP = 10  # Overlap to catch ads at chunk boundaries

    has_speakers = any("speaker" in seg for seg in segments)

    for i in range(0, len(segments), CHUNK_SIZE - OVERLAP):
        chunk = segments[i:i + CHUNK_SIZE]

        transcript_lines = []
        for seg in chunk:
            start = seg["start"]
            end = seg["end"]
            text = seg["text"].strip()
            speaker = seg.get("speaker", "")
            if speaker:
                transcript_lines.append(f"[{start:.1f}s - {end:.1f}s] {speaker}: {text}")
            else:
                transcript_lines.append(f"[{start:.1f}s - {end:.1f}s]: {text}")

        transcript_text = "\n".join(transcript_lines)

        chunk_start_time = chunk[0]["start"]
        chunk_end_time = chunk[-1]["end"]

        speaker_hint = ""
        if has_speakers:
            speaker_hint = "\nDifferent speakers (SPEAKER_XX) may indicate pre-recorded ads."

        prompt = f"""<|im_start|>system
You identify ad breaks in podcasts by detecting when the conversation goes OFF-TOPIC to promote products/services.

HOW TO IDENTIFY ADS:
1. First, understand the MAIN TOPIC of the episode from the conversation
2. Ads are OFF-TOPIC segments that promote external products/services (not related to the episode's subject)
3. Ads typically contain URLs ("visit site.com") or promo codes ("use code SAVE20")
4. Ads often start with phrases like "this episode is brought to you by" or come after "right after this"

WHAT IS NOT AN AD:
- Discussion of the episode's main topic, even if products are mentioned as part of the story
- The podcast promoting itself or its own social media
- Casual conversation between hosts

RULES:
1. Return segments where the hosts leave the main topic to promote external products
2. If multiple ads play back-to-back, combine into ONE segment (start of first to end of last)
3. Ad ends when hosts return to the episode's main topic

Output: [{{"start":SECONDS,"end":SECONDS}}] or [] if no ads<|im_end|>
<|im_start|>user
{transcript_text}<|im_end|>
<|im_start|>assistant
"""

        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=1024,
            verbose=False,
        )

        chunk_ads = _parse_ad_response(response)
        all_ad_segments.extend(chunk_ads)

    # Merge overlapping segments from different chunks
    if all_ad_segments:
        all_ad_segments = merge_overlapping_segments(all_ad_segments)

        # Do refinement pass on full transcript
        all_ad_segments = _refine_ad_segments(segments, all_ad_segments, model, tokenizer)

    return all_ad_segments


def _parse_ad_response(response: str) -> list[tuple[float, float]]:
    """Parse LLM response to extract ad segments."""
    ad_segments = []

    # Stop at the first end-of-turn token if present
    # Stop at end-of-turn tokens (Qwen uses <|im_end|>, Llama uses <|eot_id|>)
    for end_token in ["<|im_end|>", "<|eot_id|>"]:
        if end_token in response:
            response = response.split(end_token)[0]
            break

    # Minimum ad duration - ads shorter than 10 seconds are unlikely
    MIN_AD_DURATION_MS = 10000

    # Clean up common LLM formatting issues before parsing
    # Remove markdown code blocks
    response = re.sub(r'```json\s*', '', response)
    response = re.sub(r'```\s*', '', response)
    # Remove 's' suffix from numbers in JSON (e.g., "1527.1s" -> "1527.1")
    response = re.sub(r'(\d+\.?\d*)s([,\}\]])', r'\1\2', response)

    # Try JSON format first: [{"start": 45.0, "end": 120.5}]
    try:
        json_match = re.search(r'\[\s*\{[\s\S]*?\}\s*\]', response)
        if json_match:
            json_str = json_match.group()
            segments_data = json.loads(json_str)
            for seg in segments_data:
                start_ms = float(seg["start"]) * 1000
                end_ms = float(seg["end"]) * 1000
                if end_ms - start_ms >= MIN_AD_DURATION_MS:
                    ad_segments.append((start_ms, end_ms))
            return ad_segments
    except (json.JSONDecodeError, KeyError, TypeError):
        pass  # Try fallback format

    # Fallback: try bracket format like [296.7s - 298.5s]
    bracket_matches = re.findall(r'\[(\d+\.?\d*)s?\s*-\s*(\d+\.?\d*)s?\]', response)
    if bracket_matches:
        for start_str, end_str in bracket_matches:
            start_ms = float(start_str) * 1000
            end_ms = float(end_str) * 1000
            if end_ms - start_ms >= MIN_AD_DURATION_MS:
                ad_segments.append((start_ms, end_ms))
        return ad_segments

    # Empty array is valid
    if re.search(r'\[\s*\]', response):
        return []

    return ad_segments


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
        ad_segments = detect_ad_segments(transcript)

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
