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
LLM_MODEL = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
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
        # Add <|eot_id|> to the EOS token IDs so generation stops properly
        # Llama 3 uses 128009 for <|eot_id|> but it's not in eos_token_ids by default
        eot_id = _llm_tokenizer.encode("<|eot_id|>", add_special_tokens=False)[0]
        if hasattr(_llm_tokenizer, 'eos_token_ids'):
            _llm_tokenizer.eos_token_ids.add(eot_id)
            logger.info(f"[LLM] Added {eot_id} to eos_token_ids: {_llm_tokenizer.eos_token_ids}")
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
        diarization = pipeline(wav_path)

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
    Use LLM to analyze transcript and identify advertisement segments.

    Returns list of (start_ms, end_ms) tuples for ad segments.
    """
    model, tokenizer = get_llm()

    segments = transcript.get("segments", [])
    if not segments:
        return []

    # Build transcript text with timestamps and speaker labels
    transcript_lines = []
    has_speakers = any("speaker" in seg for seg in segments)
    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        text = seg["text"].strip()
        speaker = seg.get("speaker", "")
        if speaker:
            transcript_lines.append(f"[{start:.1f}s - {end:.1f}s] {speaker}: {text}")
        else:
            transcript_lines.append(f"[{start:.1f}s - {end:.1f}s]: {text}")

    transcript_text = "\n".join(transcript_lines)

    # Log transcript size
    logger.info(f"[AD_DETECTION] Transcript has {len(segments)} segments, {len(transcript_text)} chars")

    # Check if transcript is too long and needs chunking
    # Llama 3 8B has 8K context, but with 4-bit quantization we should be conservative
    # ~4 chars per token, system prompt is ~2500 chars, leave room for response
    MAX_TRANSCRIPT_CHARS = 12000

    if len(transcript_text) > MAX_TRANSCRIPT_CHARS:
        logger.info(f"[AD_DETECTION] Transcript too long ({len(transcript_text)} chars), processing in chunks")
        return _detect_ads_chunked(segments, model, tokenizer)

    # Build system prompt - include speaker hints if diarization was done
    speaker_hint = ""
    if has_speakers:
        speaker_hint = """
SPEAKER PATTERNS (transcript has speaker labels like SPEAKER_00, SPEAKER_01):
- Pre-recorded ads often come from a DIFFERENT speaker than the main host(s)
- If a new speaker appears briefly with promotional content, it's likely an ad
- Host-read ads come from the main speaker but contain ad language
"""

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You identify ads in podcast transcripts. Be aggressive - when in doubt, mark it as an ad.

ADS contain ANY of: sponsor names, promo codes, URLs, "brought to you by", "use code", "percent off", "go to [website].com", product names, brand names, ".com", product pitches, service promotions

CRITICAL AD TRIGGERS - when you see these, mark from that point until content resumes:
- "right after this ad", "right after this", "after these messages"
- "we'll be right back", "quick break", "word from our sponsors"
- Any mention of promo codes, discounts, or website URLs

COMMON AD PATTERNS:
- PRE-ROLL ADS: Ads right after show intro (often different speakers than the host)
- MID-ROLL ADS: Ads in the middle of the episode
- Multiple back-to-back ads from different speakers
{speaker_hint}
NOT ADS: show intros (just the intro itself), host banter about the episode topic, interviews, discussions

Output ONLY valid JSON: [{{"start":seconds,"end":seconds}}]
Empty if no ads: []<|eot_id|><|start_header_id|>user<|end_header_id|>

{transcript_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

["""

    # Log prompt length
    logger.info(f"[AD_DETECTION] Full prompt length: {len(prompt)} chars")

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=1024,
        verbose=False,
    )
    # Prepend the '[' we used to prime the response
    response = "[" + response

    # Log the raw LLM response
    logger.info(f"[AD_DETECTION] === LLM RAW RESPONSE ===")
    logger.info(response)

    return _parse_ad_response(response)


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
        logger.info(f"[AD_DETECTION] Processing chunk {i//CHUNK_SIZE + 1}: {chunk_start_time:.1f}s - {chunk_end_time:.1f}s")

        speaker_hint = ""
        if has_speakers:
            speaker_hint = "\nDifferent speakers (SPEAKER_XX) may indicate pre-recorded ads."

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You identify ads in podcast transcripts. Be aggressive - when in doubt, mark it as an ad.

ADS contain ANY of: sponsor names, promo codes, URLs, ".com", "use code", "percent off", product/brand names
CRITICAL: "right after this ad", "we'll be right back" - mark from here until content resumes{speaker_hint}

NOT ADS: show intros, interviews, discussions

Output ONLY valid JSON: [{{"start":seconds,"end":seconds}}]
Empty if no ads: []<|eot_id|><|start_header_id|>user<|end_header_id|>

{transcript_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

["""

        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=512,
            verbose=False,
        )
        # Prepend the '[' we used to prime the response
        response = "[" + response

        logger.info(f"[AD_DETECTION] Chunk response: {response[:200]}")

        chunk_ads = _parse_ad_response(response)
        all_ad_segments.extend(chunk_ads)

    # Merge overlapping segments from different chunks
    # Use a larger gap tolerance since chunk boundaries may split ads
    if all_ad_segments:
        all_ad_segments = merge_overlapping_segments(all_ad_segments)
        logger.info(f"[AD_DETECTION] After merging: {len(all_ad_segments)} ad segments")

    return all_ad_segments


def _parse_ad_response(response: str) -> list[tuple[float, float]]:
    """Parse LLM response to extract ad segments."""
    ad_segments = []

    # Stop at the first end-of-turn token if present
    if "<|eot_id|>" in response:
        response = response.split("<|eot_id|>")[0]

    # Minimum ad duration - ads shorter than 10 seconds are unlikely
    MIN_AD_DURATION_MS = 10000

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
            logger.info(f"[AD_DETECTION] Parsed {len(ad_segments)} ad segments from JSON response (filtered by min duration)")
            return ad_segments
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"[AD_DETECTION] JSON parse failed: {e}, trying fallback format")

    # Fallback: try bracket format like [296.7s - 298.5s]
    bracket_matches = re.findall(r'\[(\d+\.?\d*)s?\s*-\s*(\d+\.?\d*)s?\]', response)
    if bracket_matches:
        for start_str, end_str in bracket_matches:
            start_ms = float(start_str) * 1000
            end_ms = float(end_str) * 1000
            if end_ms - start_ms >= MIN_AD_DURATION_MS:
                ad_segments.append((start_ms, end_ms))
        logger.info(f"[AD_DETECTION] Parsed {len(ad_segments)} ad segments from bracket format (filtered by min duration)")
        return ad_segments

    # Empty array is valid
    if re.search(r'\[\s*\]', response):
        logger.info("[AD_DETECTION] Empty array - no ads detected")
        return []

    logger.warning(f"[AD_DETECTION] Could not parse response: {response[:500]}")
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

        # Log the full transcript with speaker labels
        logger.info(f"[EPISODE {episode_id}] === TRANSCRIPT ===")
        for seg in transcript.get("segments", []):
            speaker = seg.get("speaker", "")
            if speaker:
                logger.info(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {speaker}: {seg['text'].strip()}")
            else:
                logger.info(f"[{seg['start']:.1f}s - {seg['end']:.1f}s]: {seg['text'].strip()}")

        # Step 3: Detect ads
        report_status(callback_url, episode_id, "processing", "analyzing")
        self.update_state(
            state="ANALYZING",
            meta={"episode_id": episode_id, "step": "detecting advertisements"},
        )
        ad_segments = detect_ad_segments(transcript)

        # Log detected ad segments
        logger.info(f"[EPISODE {episode_id}] === DETECTED AD SEGMENTS ===")
        if ad_segments:
            for i, (start_ms, end_ms) in enumerate(ad_segments):
                logger.info(f"  Ad {i+1}: {start_ms/1000:.1f}s - {end_ms/1000:.1f}s (duration: {(end_ms-start_ms)/1000:.1f}s)")
        else:
            logger.info("  No ads detected")

        # Save transcripts to temp folder for debugging
        original_transcript_path = Path(temp_dir) / f"{episode_id}_transcript_original.txt"
        cleaned_transcript_path = Path(temp_dir) / f"{episode_id}_transcript_cleaned.txt"

        def is_in_ad(start_s, end_s, ad_segs):
            """Check if a segment overlaps with any ad segment."""
            start_ms = start_s * 1000
            end_ms = end_s * 1000
            for ad_start, ad_end in ad_segs:
                if start_ms < ad_end and end_ms > ad_start:
                    return True
            return False

        with open(original_transcript_path, "w") as f:
            f.write(f"=== ORIGINAL TRANSCRIPT (Episode {episode_id}) ===\n\n")
            for seg in transcript.get("segments", []):
                speaker = seg.get("speaker", "")
                prefix = f"{speaker}: " if speaker else ""
                f.write(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {prefix}{seg['text'].strip()}\n")

        with open(cleaned_transcript_path, "w") as f:
            f.write(f"=== CLEANED TRANSCRIPT (Episode {episode_id}) ===\n")
            f.write(f"=== Ad segments removed: {len(ad_segments)} ===\n\n")
            for i, (start_ms, end_ms) in enumerate(ad_segments):
                f.write(f"  [AD {i+1}] {start_ms/1000:.1f}s - {end_ms/1000:.1f}s (duration: {(end_ms-start_ms)/1000:.1f}s)\n")
            f.write("\n--- Kept segments ---\n\n")
            for seg in transcript.get("segments", []):
                speaker = seg.get("speaker", "")
                prefix = f"{speaker}: " if speaker else ""
                if is_in_ad(seg["start"], seg["end"], ad_segments):
                    f.write(f"[REMOVED] [{seg['start']:.1f}s - {seg['end']:.1f}s] {prefix}{seg['text'].strip()}\n")
                else:
                    f.write(f"[KEPT]    [{seg['start']:.1f}s - {seg['end']:.1f}s] {prefix}{seg['text'].strip()}\n")

        logger.info(f"[EPISODE {episode_id}] Saved transcripts to {temp_dir}")

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
        # DEBUG: Keep temp directory for inspection
        logger.info(f"[EPISODE {episode_id}] Temp directory preserved at: {temp_dir}")
        logger.info(f"[EPISODE {episode_id}]   - Input: {input_path}")
        logger.info(f"[EPISODE {episode_id}]   - Output: {output_path}")
        # shutil.rmtree(temp_dir, ignore_errors=True)  # Commented out for debugging


if __name__ == "__main__":
    app.start()
