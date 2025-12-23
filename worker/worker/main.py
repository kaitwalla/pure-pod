import json
import logging
import os
import re
import shutil
import tempfile
import warnings
from datetime import datetime
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
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
HF_TOKEN = os.environ.get("HF_TOKEN")  # Required for pyannote
WORKER_API_KEY = os.environ.get("WORKER_API_KEY")  # Required for upload authentication

WHISPER_MODEL = "mlx-community/distil-whisper-large-v3"
LLM_MODEL = "mlx-community/Qwen2.5-14B-Instruct-4bit"
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

# Directory for storing cleaned audio files that failed to upload
PENDING_UPLOADS_DIR = Path(os.environ.get("PENDING_UPLOADS_DIR", "/tmp/pending_uploads"))

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
    # Redis connection resilience for remote connections
    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
    broker_connection_max_retries=10,
    broker_pool_limit=1,  # Single connection since we only process one task at a time
    broker_heartbeat=30,  # Send heartbeat every 30 seconds
    broker_transport_options={
        "socket_keepalive": True,
        "socket_connect_timeout": 30,
        "socket_timeout": 30,
    },
    result_backend_transport_options={
        "socket_keepalive": True,
        "socket_connect_timeout": 30,
        "socket_timeout": 30,
    },
)

# Cache loaded models at module level
_llm_model = None
_llm_tokenizer = None
_diarization_pipeline = None
_embedding_model = None
_startup_validated = False

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, good quality embeddings


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


def get_embedding_model():
    """Lazy-load and cache the sentence embedding model."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"[EMBEDDINGS] Loading sentence transformer model: {EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("[EMBEDDINGS] Model loaded successfully")
    return _embedding_model


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
        # Limit max speakers - podcasts can have many guests plus ad voices
        diarization = pipeline(wav_path, max_speakers=15)

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

    # === PASS 2: Find incongruous content (LLM-based) ===
    pass2_ads = _pass2_find_incongruous_content_llm(segments, episode_context, pass1_ads, model, tokenizer)

    # Combine pass 1 and pass 2 results
    all_ads = pass1_ads + pass2_ads
    if all_ads:
        all_ads = merge_overlapping_segments(all_ads)

    # === PASS 3: Verify and refine boundaries ===
    # DISABLED - testing if this is causing issues
    # if all_ads:
    #     all_ads = _pass3_verify_boundaries(segments, episode_context, all_ads, model, tokenizer)

    # === PASS 4: Deterministic trim at episode content ===
    # DISABLED - was too aggressive and trimming valid ads
    # if all_ads:
    #     all_ads = _pass4_trim_at_episode_content(segments, all_ads)

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

        # Get context: 120 seconds before, 300 seconds after
        # Ad blocks can be 3-4+ minutes long (multiple ads back-to-back)
        # Need more lookback because ad intros ("brought to you by") can be far before the URL
        context_start = max(0, group_start - 120)
        context_end = min(segments[-1]["end"], group_end + 300)

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
- "brought to you by", "sponsored by", "speaking of which", "a word from our sponsor"
- Sudden topic change from episode content to product/service promotion
- Speaker change to someone reading an ad
- PRE-ROLL ADS: If the indicator is near the start of the episode, the ad may start at the VERY BEGINNING (0s)
- Look for the FIRST mention of the product/brand being advertised - that's where the ad starts

AD END - look forwards from the last indicator until you find a CLEAR return to episode content:
- Include ALL ad content: final call-to-action, promo code reminder, "thanks to X", closing pitch
- The ad ends when the host CLEARLY returns to the episode topic (not just transition words)
- Do NOT end at "anyway", "so", "alright" - these are often STILL PART of the ad transition

CLEAR signs the episode has resumed (ad has ended):
- Host says "Welcome back", "We're back", "Alright, back to..."
- Host addresses the guest by name and asks a question about the episode topic
- **IMPORTANT**: Discussion returns to topics from the EPISODE TITLE or DESCRIPTION above - this is your best signal!
- Host references something discussed BEFORE the ad break ("So as we were saying...", "To continue...")
- The conversation topic matches keywords from the episode description
- Listener Q&A segments: "question from a listener", "our next/last question", "listener named X", "mailbag", "Q&A"
- Interview segments resuming: "so tell me about", "what do you think about", "how did you"

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
                if end_ms - start_ms >= 15000:  # Min 15 seconds
                    ad_segments.append((start_ms, end_ms))
        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback: use indicator group boundaries with padding
            start_ms = max(0, (group_start - 30)) * 1000
            end_ms = (group_end + 10) * 1000
            ad_segments.append((start_ms, end_ms))

    return ad_segments


def _pass2_find_incongruous_content_llm(
    segments: list,
    episode_context: str,
    existing_ads: list[tuple[float, float]],
    model,
    tokenizer
) -> list[tuple[float, float]]:
    """
    Pass 2 (LLM version - DEPRECATED, kept for reference):
    Find content that is incongruous with both:
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

    HIGH_CONFIDENCE = 0.8  # Include these directly
    MEDIUM_CONFIDENCE = 0.5  # Include if clustered with others
    CLUSTER_GAP_SECONDS = 20  # Max gap between detections to consider them clustered
    MIN_AD_DURATION_MS = 15000  # Minimum 15 seconds

    prompt = f"""<|im_start|>system
You are finding advertisements that were MISSED in a first pass.

EPISODE INFO (this is what the episode is actually about):
{episode_context}

Some segments are already marked as [ALREADY MARKED AS AD] - ignore those.

Find UNMARKED segments that are INCONGRUOUS with the episode content - they don't belong.

KEY SIGNAL: Does this segment fit the episode topic above? If NOT, it's probably an ad.

PRE-ROLL ADS (at the very beginning):
- Check the FIRST 3-5 MINUTES carefully - pre-roll ads often appear before the episode intro
- Pre-roll ads may start abruptly without "brought to you by" - just straight into the pitch
- Look for: product pitches, brand mentions, promotional content BEFORE the hosts introduce the episode
- The episode usually starts with: theme music, host introduction, episode topic overview

BRAND ADS (no URLs/promo codes - just brand mentions):
- Sudden pivot to talking about a product/service/brand unrelated to episode topic
- Descriptive language about a product ("crispy", "refreshing", "delicious", "smooth", "premium")
- Benefits/features of a product that has nothing to do with the episode
- "I love [brand]", "I've been using [product]", "[Brand] has been great"
- Any product/service pitch that interrupts the natural flow of conversation

IMPORTANT: Ads often REPEAT multiple times per episode. Each instance is still an ad!
- The same product pitch appearing 2-3 times = 2-3 separate ads to mark
- Mark EACH instance separately

How to identify brand ads:
1. **READ THE EPISODE TITLE AND DESCRIPTION CAREFULLY** - what is this episode actually about?
2. Content that matches the title/description = EPISODE CONTENT, not an ad
3. Find segments where the topic SHIFTS AWAY from the title/description to something unrelated (a product, brand, service)
4. These shifts are ads, even without URLs or "use code X"
5. When in doubt, check: does this segment relate to the episode title/description? If yes, it's NOT an ad.

Examples of what to find:
- "Speaking of comfort, I've been sleeping so much better on my new [mattress brand]..."
- "You know what I love? [Food brand]. So crispy and delicious..."
- "I've been using [app/service] and it's changed how I [do thing]..."
- Any segment praising a product/brand that has nothing to do with the episode topic

Examples of what is NOT an ad:
- Discussion related to the episode topic (even if mentioning brands as examples relevant to the topic)
- The podcast promoting its own social media or Patreon
- Hosts casually mentioning something they bought in context of the conversation topic

For each ad found, provide a confidence score (0.0-1.0):
- 0.9-1.0: Definitely an ad (clear brand pitch unrelated to episode)
- 0.7-0.9: Likely an ad (product praise that seems out of place)
- 0.5-0.7: Possible ad (unclear if organic mention or sponsored)

Return JSON array: [{{"start":SECONDS,"end":SECONDS,"confidence":0.0-1.0}}]
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

            # Collect all valid detections with confidence
            all_detections = []
            for seg in segments_data:
                start_ms = float(seg["start"]) * 1000
                end_ms = float(seg["end"]) * 1000
                confidence = float(seg.get("confidence", 1.0))
                if end_ms - start_ms >= MIN_AD_DURATION_MS:
                    all_detections.append((start_ms, end_ms, confidence))

            # Sort by start time
            all_detections.sort(key=lambda x: x[0])

            # Process detections: high confidence always included,
            # medium confidence included if clustered
            for i, (start_ms, end_ms, confidence) in enumerate(all_detections):
                if confidence >= HIGH_CONFIDENCE:
                    new_ads.append((start_ms, end_ms))
                elif confidence >= MEDIUM_CONFIDENCE:
                    # Check if clustered with neighbors
                    has_neighbor = False
                    gap_ms = CLUSTER_GAP_SECONDS * 1000

                    # Check previous detection
                    if i > 0:
                        prev_end = all_detections[i-1][1]
                        if start_ms - prev_end <= gap_ms:
                            has_neighbor = True

                    # Check next detection
                    if i < len(all_detections) - 1:
                        next_start = all_detections[i+1][0]
                        if next_start - end_ms <= gap_ms:
                            has_neighbor = True

                    if has_neighbor:
                        new_ads.append((start_ms, end_ms))
                # Below MEDIUM_CONFIDENCE: discard

    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    return new_ads


def _pass2_find_incongruous_content(
    segments: list,
    episode_context: str,
    existing_ads: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """
    Pass 2 (Embedding version): Find content that is semantically different from the episode topic.

    Uses sentence embeddings to:
    1. Create a "topic anchor" from the episode title/description
    2. Score each segment by similarity to the topic
    3. Find contiguous low-similarity regions as candidate ads

    This catches ads that don't have explicit markers but are off-topic.
    """
    embedding_model = get_embedding_model()

    if not segments:
        return []

    # Skip segments already marked as ads
    def is_in_existing_ad(start_s, end_s):
        start_ms = start_s * 1000
        end_ms = end_s * 1000
        for ad_start, ad_end in existing_ads:
            if start_ms < ad_end and end_ms > ad_start:
                return True
        return False

    # Create topic anchor embedding from episode context
    if not episode_context or episode_context == "No episode metadata available.":
        logger.warning("[EMBEDDINGS] No episode context available, skipping Pass 2")
        return []

    topic_embedding = embedding_model.encode([episode_context])[0]

    # Group segments into chunks for more stable embeddings
    # Individual segments can be too short for meaningful comparison
    CHUNK_SIZE = 5  # Number of segments per chunk
    MIN_SIMILARITY_THRESHOLD = 0.15  # Below this = likely off-topic
    LOW_SIMILARITY_THRESHOLD = 0.25  # Below this = possibly off-topic
    MIN_AD_DURATION_MS = 15000  # Minimum 15 seconds

    # Build chunks
    chunks = []
    for i in range(0, len(segments), CHUNK_SIZE):
        chunk_segments = segments[i:i + CHUNK_SIZE]

        # Skip if most of chunk is already marked as ad
        ad_count = sum(1 for seg in chunk_segments if is_in_existing_ad(seg["start"], seg["end"]))
        if ad_count > len(chunk_segments) / 2:
            continue

        chunk_text = " ".join(seg["text"].strip() for seg in chunk_segments)
        chunk_start = chunk_segments[0]["start"]
        chunk_end = chunk_segments[-1]["end"]

        chunks.append({
            "text": chunk_text,
            "start": chunk_start,
            "end": chunk_end,
            "segments": chunk_segments,
        })

    if not chunks:
        return []

    # Generate embeddings for all chunks
    chunk_texts = [c["text"] for c in chunks]
    chunk_embeddings = embedding_model.encode(chunk_texts)

    # Calculate similarity to topic anchor
    for i, chunk in enumerate(chunks):
        similarity = cosine_similarity([chunk_embeddings[i]], [topic_embedding])[0][0]
        chunk["similarity"] = similarity

    # Log similarity scores for debugging
    logger.info("[EMBEDDINGS] Chunk similarities to episode topic:")
    for chunk in chunks[:10]:  # Log first 10 for brevity
        logger.info(f"  [{chunk['start']:.1f}s - {chunk['end']:.1f}s] sim={chunk['similarity']:.3f}: {chunk['text'][:80]}...")

    # Find low-similarity regions
    low_sim_chunks = []
    for chunk in chunks:
        if chunk["similarity"] < LOW_SIMILARITY_THRESHOLD:
            low_sim_chunks.append(chunk)

    if not low_sim_chunks:
        logger.info("[EMBEDDINGS] No low-similarity chunks found")
        return []

    # Group consecutive low-similarity chunks
    ad_regions = []
    current_region = [low_sim_chunks[0]]

    for chunk in low_sim_chunks[1:]:
        # Check if this chunk is close to the previous one (within 30 seconds)
        if chunk["start"] - current_region[-1]["end"] < 30:
            current_region.append(chunk)
        else:
            ad_regions.append(current_region)
            current_region = [chunk]
    ad_regions.append(current_region)

    # Convert regions to ad segments
    new_ads = []
    for region in ad_regions:
        start_ms = region[0]["start"] * 1000
        end_ms = region[-1]["end"] * 1000

        # Calculate average similarity for the region
        avg_similarity = sum(c["similarity"] for c in region) / len(region)

        # Only include if below threshold and long enough
        if avg_similarity < LOW_SIMILARITY_THRESHOLD and end_ms - start_ms >= MIN_AD_DURATION_MS:
            logger.info(f"[EMBEDDINGS] Found off-topic region: {start_ms/1000:.1f}s - {end_ms/1000:.1f}s (avg_sim={avg_similarity:.3f})")
            new_ads.append((start_ms, end_ms))

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
   - Are they transition phrases that are part of the ad? ("check them out", "thanks again to", "alright")
   - Is the ACTUAL return to episode content further down? Look for: guest questions, topic discussion, "welcome back"
   - If yes, extend the ad to include everything up to the real episode content

3. ENDED TOO LATE - Adjust if we included clear episode content:
   - **KEY**: Does the content match the EPISODE TITLE or DESCRIPTION above? If yes, it's episode content!
   - Listener Q&A: "question from a listener", "our next/last question", "listener named X"
   - Interview resuming: host asks guest a question about the episode topic
   - A single transition word ("anyway", "so", "alright") is NOT enough to trim
   - But content matching the episode description IS enough to trim

BIAS: When in doubt, keep the full ad segment. It's better to remove a bit of transition than leave ad content.

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
                            if new_end_ms - new_start_ms >= 15000:  # Min 15 seconds
                                adjusted_ads[i] = (new_start_ms, new_end_ms)
                            break

                # Re-merge in case adjustments created overlaps
                return merge_overlapping_segments(adjusted_ads)
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    return ad_segments


def _pass4_trim_at_episode_content(
    segments: list,
    ad_segments: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """
    Pass 4: Deterministically trim ads that include clear episode content.

    Checks both directions:
    - START: If ad begins with episode content, move start forward
    - END: If ad ends with episode content, move end backward
    """
    # Patterns that indicate EPISODE CONTENT (not ads)
    # NOTE: Only include patterns that are UNAMBIGUOUS - would never appear in an ad
    episode_content_patterns = [
        # Listener Q&A segments - very specific, won't appear in ads
        r'\b(first|next|last|final)\s+question\b',
        r'\bquestion\s+(from|comes?\s+from)\b',
        r'\blistener\s+named\b',
        r'\blistener\s+in\s+\w+\b',  # "listener in Indianapolis"
        r'\bwrites\s+in\b',  # "John writes in"
        r'\bmailbag\b',
        r'\bq\s*&\s*a\b',
        r'\b(she|he)\s+asked\b',  # "she asked about"
        r'\b(she|he)\'s\s+wondered\b',  # "she's wondered about"
        r'\bwondered\s+about\b',
        # Listener speaking (first person statements about hobbies, experiences)
        r'\bi\s+am\s+a\s+big\b',  # "I am a big musical lover"
        r'\bi\s+go\s+to\s+(live|the)\b',  # "I go to live shows"
        r'\bi\'ve\s+(always|been|never)\b',  # "I've always wondered"
        # Interview/discussion continuation
        r'\b(she|he|they)\'s\s+(sort\s+of\s+)?(singing|playing|doing|saying|showing|wondered)\b',
        r'\bthat\s+is\s+a\s+really\s+(incredible|amazing|great)\b',
        r'\bworth\s+the\s+price\b',
        r'\bprice\s+of\s+admission\b',
        # Story/narrative continuation
        r'\bso\s+(she|he|they)\s+(said|did|went|started)\b',
        r'\band\s+then\s+(she|he|they)\b',
        r'\band\s+(she|he|they)\'s\b',  # "And she's wondered"
        r'\bwhile\s+(she|he|they)\b',
        # Emotional reactions (audience, etc.)
        r'\bi\s+love\s+you\b',
        # Return from break indicators
        r'\bwelcome\s+back\b',
        r'\bwe\'?re\s+back\b',
        r'\bback\s+to\s+(the|our)\b',
        r'\bas\s+(we|i)\s+was\s+saying\b',
        r'\bto\s+continue\b',
    ]

    # Patterns that indicate AD/PROMO content (should be removed)
    ad_content_patterns = [
        # Ad intro phrases (start of ads)
        r'\bbrought\s+to\s+you\s+by\b',
        r'\bsponsored\s+by\b',
        r'\btoday\'?s\s+sponsor\b',
        r'\bsupport\s+(for\s+)?(this|the)\s+(show|podcast|episode)\b',
        r'\bword\s+from\s+(our\s+)?sponsor\b',
        r'\blet\s+me\s+tell\s+you\s+about\b',
        r'\bintroducing\b',
        r'\bhave\s+you\s+(ever\s+)?(tried|heard|wondered)\b',
        r'\bever\s+wish(ed)?\b',
        r'\btired\s+of\b',
        r'\bstruggling\s+with\b',
        # URLs and calls to action
        r'\b\w+\.(com|org|co|net)\b',  # URLs
        r'\bgo\s+to\s+\w+\b',  # "go to site"
        r'\bvisit\s+\w+\b',  # "visit site"
        r'\bcheck\s+out\s+\w+\b',  # "check out site"
        r'\bhead\s+(to|over)\b',  # "head to"
        r'\bsign\s+up\b',  # "sign up"
        r'\bclick\s+(the\s+)?link\b',  # "click the link"
        # Promo codes and specific offers
        r'\bpromo\s*code\b',
        r'\buse\s+code\b',
        r'\bdiscount\s+code\b',
        r'\b\d+%\s*off\b',  # "50% off"
        r'\bfree\s+(trial|for\s+\d+)\b',  # "free trial" or "free for 30 days"
        r'\bfree\s+shipping\b',
        r'\bno\s+credit\s+card\b',  # "no credit card required"
        r'\bcancel\s+anytime\b',
        r'\bmoney[\s-]back\b',
        # Thanks/closing sponsor mentions
        r'\bthanks\s+to\s+\w+\s+for\s+sponsor',
        # Donation/membership asks
        r'\bplease\s+(consider\s+)?(support|donat)',
        r'\bgive\s+today\b',
        r'\bdonate\s+(now|today)\b',
        r'\bbecome\s+a\s+\w+\s*(plus\s+)?member\b',
        r'\b(slate|patreon|substack)\s*plus\s*(member|$)',
        r'\bjoin\s+(us\s+)?(on\s+)?(patreon|substack)\b',
    ]

    combined_episode_pattern = '|'.join(episode_content_patterns)
    combined_ad_pattern = '|'.join(ad_content_patterns)

    trimmed_ads = []

    for ad_start_ms, ad_end_ms in ad_segments:
        # Get segments within this ad
        ad_segs = [
            seg for seg in segments
            if seg["start"] * 1000 >= ad_start_ms - 1000  # Small tolerance
            and seg["end"] * 1000 <= ad_end_ms + 1000
        ]

        if not ad_segs:
            trimmed_ads.append((ad_start_ms, ad_end_ms))
            continue

        # Classify each segment
        classified = []
        for seg in ad_segs:
            text = seg["text"]
            is_episode = bool(re.search(combined_episode_pattern, text, re.IGNORECASE))
            is_ad = bool(re.search(combined_ad_pattern, text, re.IGNORECASE))
            classified.append({
                "seg": seg,
                "is_episode": is_episode,
                "is_ad": is_ad,
            })

        # Find first segment that's clearly an ad (for trimming start)
        first_ad_idx = None
        for i, c in enumerate(classified):
            if c["is_ad"] and not c["is_episode"]:
                first_ad_idx = i
                break

        # Find last segment that's clearly an ad (for trimming end)
        last_ad_idx = None
        for i in range(len(classified) - 1, -1, -1):
            if classified[i]["is_ad"] and not classified[i]["is_episode"]:
                last_ad_idx = i
                break

        new_start_ms = ad_start_ms
        new_end_ms = ad_end_ms

        # Strategy: Find the first and last segments that are clearly ads,
        # then find any episode content before/after and trim to exclude it

        if last_ad_idx is not None:
            # Find FIRST episode content segment AFTER the last ad segment
            for c in classified[last_ad_idx + 1:]:
                if c["is_episode"] and not c["is_ad"]:
                    # Trim end to just after the last ad segment
                    new_end_ms = classified[last_ad_idx]["seg"]["end"] * 1000
                    break

        if first_ad_idx is not None:
            # Find LAST episode content segment BEFORE the first ad segment
            for c in reversed(classified[:first_ad_idx]):
                if c["is_episode"] and not c["is_ad"]:
                    # Trim start to the first ad segment
                    new_start_ms = classified[first_ad_idx]["seg"]["start"] * 1000
                    break

        # Also trim consecutive episode content from the very edges
        # This handles cases where there's no clear ad marker but episode content at edges
        for c in classified:
            if c["is_episode"] and not c["is_ad"]:
                new_start_ms = max(new_start_ms, c["seg"]["end"] * 1000)
            else:
                break

        for c in reversed(classified):
            if c["is_episode"] and not c["is_ad"]:
                new_end_ms = min(new_end_ms, c["seg"]["start"] * 1000)
            else:
                break

        # Only keep if still a valid ad (at least 10 seconds)
        if new_end_ms - new_start_ms >= 10000:
            trimmed_ads.append((new_start_ms, new_end_ms))

    return trimmed_ads


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
        r'\b\w+\.gov\b',  # something.gov
        r'\b\w+\.net\b',  # something.net
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
        r'\bapply\s+now\b',  # "apply now"
        r'\bbook\s+now\b',  # "book now"
        r'\bsave\s+(over\s+)?\$?\d+',  # "save $200" or "save over $200"
        r'\bbonuses?\s+up\s+to\b',  # "bonuses up to $50,000"
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
        r'\blimited\s+time\b',  # "limited time"
        r'\bspecial\s+offer\b',  # "special offer"
        r'\bexclusive\s+(offer|deal|discount)',  # "exclusive offer"
        r'\brisk[\s-]free\b',  # "risk-free"
        r'\bwhile\s+supplies\s+last\b',  # "while supplies last"
        # Retail/availability language
        r'\bavailab(le|ility)\b',  # "available" or "availability"
        r'\bin\s+stores\s+(now|today|nationwide)',  # "in stores now"
        r'\bfind\s+(it|us|them)\s+(at|in)\b',  # "find it at"
        r'\bnow\s+available\b',  # "now available"
        r'\bget\s+(it|yours)\s+(now|today|at)\b',  # "get yours today"
        r'\bfor\s+a\s+limited\s+time\b',  # "for a limited time"
        # Ad break transition phrases (indicates ad is starting/ending)
        r'\bafter\s+(the\s+)?break\b',  # "after the break" / "after break"
        r'\bbefore\s+(the\s+)?break\b',  # "before the break"
        r'\bwhen\s+we\s+(come|get)\s+back\b',  # "when we come back" / "when we get back"
        r'\bwe\'?ll\s+be\s+right\s+back\b',  # "we'll be right back"
        r'\bright\s+after\s+this\b',  # "right after this"
        r'\bstay\s+(tuned|with\s+us)\b',  # "stay tuned" / "stay with us"
        r'\bafter\s+these\s+messages\b',  # "after these messages"
        r'\btake\s+a\s+(quick\s+)?break\b',  # "take a break" / "take a quick break"
        r'\bquick\s+break\b',  # "quick break"
        r'\bshort\s+break\b',  # "short break"
        r'\blet\'?s\s+take\s+a\s+break\b',  # "let's take a break"
        # Sponsor phrases (strong ad indicators)
        r'\bbrought\s+to\s+you\s+by\b',  # "brought to you by"
        r'\bsponsored\s+by\b',  # "sponsored by"
        r'\bsupport(ed)?\s+(for\s+)?(this\s+)?(show|podcast|episode)\s+(comes?\s+from|is\s+brought)',  # "support for this show comes from"
        r'\btoday\'?s\s+sponsor\b',  # "today's sponsor"
        r'\bour\s+sponsor(s)?\b',  # "our sponsor(s)"
        r'\bthis\s+(episode|show|podcast)\s+is\s+(brought|sponsored)',  # "this episode is brought/sponsored"
        r'\blet\s+me\s+tell\s+you\s+about\b',  # "let me tell you about" (host-read intro)
        r'\bword\s+from\s+(our\s+)?sponsor',  # "word from our sponsor"
        r'\bthanks\s+to\s+\w+\s+for\s+sponsor',  # "thanks to X for sponsoring"
        r'\bmade\s+possible\s+by\b',  # "made possible by"
        r'\bquick\s+(word|message|break)\s+from',  # "quick word from"
        r'\bpresented\s+by\b',  # "presented by"
        r'\bpowered\s+by\b',  # "powered by"
        r'\bin\s+partnership\s+with\b',  # "in partnership with"
        r'\bpartnered\s+with\b',  # "partnered with"
        r'\bshoutout\s+to\b',  # "shoutout to" (sponsor mention)
        r'\bbig\s+thanks\s+to\b',  # "big thanks to"
        r'\bspecial\s+thanks\s+to\b',  # "special thanks to"
        r'\bwant\s+to\s+thank\b',  # "want to thank"
        r'\bgotta\s+thank\b',  # "gotta thank"
        r'\bhuge\s+thanks\s+to\b',  # "huge thanks to"
        # Legal disclaimers (VERY reliable - almost never in regular content)
        r'\btaxes\s+(and|&)\s+fees\b',  # "taxes and fees"
        r'\brestrictions?\s+appl(y|ies)\b',  # "restrictions apply"
        r'\bterms\s+(and|&)\s+conditions\b',  # "terms and conditions"
        r'\bsee\s+(website|site|store)\s+for\s+details\b',  # "see website for details"
        r'\boffer\s+(valid|expires|ends)\b',  # "offer valid/expires"
        r'\bsubject\s+to\s+(change|availability)\b',  # "subject to change"
        r'\bsome\s+(restrictions|exclusions)\b',  # "some restrictions"
        r'\bnot\s+available\s+in\s+all\b',  # "not available in all areas"
        r'\bguaranteed\b',  # "guaranteed" (common in ads)
        r'\brequires?\s+(at\s+least|minimum)\b',  # "requires at least"
        r'\bspeeds?\s+(may\s+)?(vary|reduce|decrease)\b',  # "speeds may vary"
        r'\bdata\s+(speeds?|limits?|caps?)\b',  # "data speeds/limits"
        r'\b(unlimited|megabits?|gigabits?|gigs?)\b',  # telecom terms
        r'\bprice\s+(lock|guarantee|protection)\b',  # "price lock"
        r'\bwon\'?t\s+go\s+up\b',  # "won't go up"
        r'\bfor\s+up\s+to\s+(\d+|one|two|three|four|five)\s+(years?|months?)\b',  # "for up to 3 years"
        r'\bup\s+to\s+(\d+|one|two|three|four|five)\s+(years?|months?)\s+guarantee',  # "up to 3 years guaranteed"
        # Pharma/regulated (hard anchors)
        r'\bask\s+your\s+(doctor|physician)',  # "ask your doctor"
        r'\bside\s+effects\s+(may\s+)?include',  # "side effects include"
        r'\b1-8(00|77|88|66)',  # 1-800 numbers
        # Alcohol ads
        r'\bdrink\s+responsibly\b',  # "drink responsibly"
        r'\bmust\s+be\s+21\b',  # "must be 21"
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


def extract_audio_url_from_tracker(url: str) -> str | None:
    """
    Extract the real audio URL from tracking/redirect URLs.

    Many podcast URLs go through tracking services like:
    - tracking.swap.fm/track/.../traffic.megaphone.fm/file.mp3
    - pdst.fm/e/...

    Returns the extracted URL or None if not a tracking URL.
    """
    # Pattern: tracking URL with embedded real URL in path
    # e.g., tracking.swap.fm/track/xxx/pscrb.fm/rss/p/traffic.megaphone.fm/FILE.mp3
    known_audio_hosts = [
        'stitcher.simplecastaudio.com',
        'simplecastaudio.com',
        'traffic.megaphone.fm',
        'megaphone.fm',
        'dts.podtrac.com',
        'chtbl.com',
        'pdst.fm',
        'anchor.fm',
        'buzzsprout.com',
        'libsyn.com',
        'soundcloud.com',
        'spreaker.com',
        'rss.art19.com',
        'arttrk.com',
        'omny.fm',
        'omnycontent.com',
        'podbean.com',
        'audioboom.com',
        'captivate.fm',
        'transistor.fm',
    ]

    for host in known_audio_hosts:
        if host in url:
            # Find the host in the URL path and extract from there
            idx = url.find(host)
            if idx > 0:
                extracted = 'https://' + url[idx:]
                # Clean up any query params that might be tracking-specific
                return extracted

    return None


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


def save_pending_upload(episode_id: str, audio_path: Path, callback_url: str, ad_segments_removed: int) -> Path:
    """
    Save a cleaned audio file for later upload retry.

    Returns the path to the saved audio file.
    """
    PENDING_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    # Copy the audio file
    pending_audio_path = PENDING_UPLOADS_DIR / f"{episode_id}.mp3"
    shutil.copy2(audio_path, pending_audio_path)

    # Save metadata
    metadata = {
        "episode_id": episode_id,
        "callback_url": callback_url,
        "ad_segments_removed": ad_segments_removed,
        "created_at": datetime.utcnow().isoformat(),
        "retry_count": 0,
    }
    metadata_path = PENDING_UPLOADS_DIR / f"{episode_id}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    logger.info(f"[PENDING] Saved episode {episode_id} for upload retry")
    return pending_audio_path


def attempt_upload(audio_path: Path, episode_id: str, callback_url: str) -> bool:
    """
    Attempt to upload a cleaned audio file.

    Returns True if successful, False otherwise.
    """
    try:
        with open(audio_path, "rb") as f:
            files = {"file": (f"{episode_id}.mp3", f, "audio/mpeg")}
            headers = {}
            if WORKER_API_KEY:
                headers["X-API-Key"] = WORKER_API_KEY
            response = requests.post(
                callback_url,
                files=files,
                data={"episode_id": episode_id},
                headers=headers,
                timeout=600,
            )
            response.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"[UPLOAD] Failed to upload episode {episode_id}: {e}")
        return False


def remove_pending_upload(episode_id: str):
    """Remove a pending upload after successful upload."""
    audio_path = PENDING_UPLOADS_DIR / f"{episode_id}.mp3"
    metadata_path = PENDING_UPLOADS_DIR / f"{episode_id}.json"

    if audio_path.exists():
        audio_path.unlink()
    if metadata_path.exists():
        metadata_path.unlink()

    logger.info(f"[PENDING] Removed pending upload for episode {episode_id}")


@app.task(name="worker.retry_pending_uploads")
def retry_pending_uploads(max_retries: int = 5) -> dict:
    """
    Retry uploading any pending cleaned audio files.

    This task can be called manually or scheduled via Celery Beat.
    """
    if not PENDING_UPLOADS_DIR.exists():
        return {"checked": 0, "succeeded": 0, "failed": 0, "abandoned": 0}

    results = {"checked": 0, "succeeded": 0, "failed": 0, "abandoned": 0}

    for metadata_path in PENDING_UPLOADS_DIR.glob("*.json"):
        results["checked"] += 1

        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"[PENDING] Failed to read metadata {metadata_path}: {e}")
            continue

        episode_id = metadata["episode_id"]
        callback_url = metadata["callback_url"]
        retry_count = metadata.get("retry_count", 0)
        audio_path = PENDING_UPLOADS_DIR / f"{episode_id}.mp3"

        if not audio_path.exists():
            logger.warning(f"[PENDING] Audio file missing for episode {episode_id}, removing metadata")
            metadata_path.unlink()
            continue

        if retry_count >= max_retries:
            logger.warning(f"[PENDING] Episode {episode_id} exceeded max retries ({max_retries}), abandoning")
            report_status(callback_url, episode_id, "failed", None, f"Upload failed after {max_retries} retries")
            remove_pending_upload(episode_id)
            results["abandoned"] += 1
            continue

        logger.info(f"[PENDING] Retrying upload for episode {episode_id} (attempt {retry_count + 1})")

        if attempt_upload(audio_path, episode_id, callback_url):
            report_status(callback_url, episode_id, "cleaned", None, None)
            remove_pending_upload(episode_id)
            results["succeeded"] += 1
            logger.info(f"[PENDING] Successfully uploaded episode {episode_id} on retry")
        else:
            # Update retry count
            metadata["retry_count"] = retry_count + 1
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)
            results["failed"] += 1

    return results


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

        # Try original URL, fall back to extracted URL if tracking service fails
        download_url = audio_url
        try:
            response = requests.get(download_url, stream=True, timeout=600, headers=headers)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            # Tracking service might be down - try extracting the real audio URL
            fallback_url = extract_audio_url_from_tracker(audio_url)
            if fallback_url and fallback_url != audio_url:
                logger.info(f"[DOWNLOAD] Tracking URL failed, trying fallback: {fallback_url}")
                download_url = fallback_url
                response = requests.get(download_url, stream=True, timeout=600, headers=headers)
                response.raise_for_status()
            else:
                raise  # Re-raise if no fallback available

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

        # Try to upload, save to pending queue on failure
        if attempt_upload(output_path, episode_id, callback_url):
            return {
                "status": "success",
                "episode_id": episode_id,
                "ad_segments_removed": len(ad_segments),
                "message": "Episode processed and uploaded successfully",
            }
        else:
            # Upload failed - save for retry
            save_pending_upload(episode_id, output_path, callback_url, len(ad_segments))
            report_status(callback_url, episode_id, "processing", "upload_pending")
            return {
                "status": "pending_upload",
                "episode_id": episode_id,
                "ad_segments_removed": len(ad_segments),
                "message": "Episode processed but upload failed - queued for retry",
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
