# PodcastPurifier ML Worker

The Worker is the ML processing node for PodcastPurifier. It runs on Apple Silicon (M1/M2/M3/M4) and uses MLX for efficient on-device transcription and ad detection.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Worker (Apple Silicon)                   │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                    Celery Worker                        ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ││
│  │  │ mlx-whisper  │  │   mlx-lm     │  │    pydub     │  ││
│  │  │ (Transcribe) │  │ (Ad Detect)  │  │ (Audio Edit) │  ││
│  │  └──────────────┘  └──────────────┘  └──────────────┘  ││
│  └─────────────────────────────────────────────────────────┘│
│                              │                               │
└──────────────────────────────┼───────────────────────────────┘
                               │ Redis
                               ▼
                    ┌─────────────────┐
                    │     Manager     │
                    │ (Task Dispatch) │
                    └─────────────────┘
```

## Processing Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Download   │───▶│  Transcribe  │───▶│ Detect Ads   │───▶│  Remove Ads  │───▶│   Upload     │
│   Audio      │    │ (Whisper)    │    │   (LLM)      │    │ (Crossfade)  │    │  to Manager  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### Pipeline Steps

1. **Download**: Fetch the original podcast audio from the source URL
2. **Transcribe**: Use distil-whisper-large-v3 for word-level transcription
3. **Detect Ads**: LLM (Llama-3-8B-Instruct-4bit) analyzes transcript for ad segments
4. **Remove Ads**: Cut ad segments with 500ms crossfade for smooth transitions
5. **Upload**: POST cleaned MP3 back to Manager's `/upload/{episode_id}` endpoint

## Models Used

| Model | Purpose | Size |
|-------|---------|------|
| `mlx-community/distil-whisper-large-v3` | Audio transcription with word timestamps | ~1.5GB |
| `mlx-community/Meta-Llama-3-8B-Instruct-4bit` | Ad segment detection from transcript | ~5.3GB |

Models are automatically downloaded on first use and cached locally.

## Requirements

- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **Python 3.11-3.13**
- **Redis** (for Celery message broker)
- **ffmpeg** (required by pydub for audio processing)

## Installation

### 1. Install System Dependencies

```bash
# Install ffmpeg (required for pydub)
brew install ffmpeg

# Install Redis (if running locally)
brew install redis
brew services start redis
```

### 2. Install Python Dependencies

Using [uv](https://github.com/astral-sh/uv) (recommended):

```bash
cd worker

# Create virtual environment and install dependencies
uv sync

# Activate the environment
source .venv/bin/activate
```

Or using pip:

```bash
cd worker
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Running the Worker

### Basic Usage

```bash
cd worker
source .venv/bin/activate

# Set Redis URL (default: redis://localhost:6379/0)
export REDIS_URL=redis://localhost:6379/0

# Start the Celery worker
celery -A worker.main worker --loglevel=info -Q audio_processing
```

### With Concurrency Control

Since ML inference is memory-intensive, limit concurrency:

```bash
# Single worker (recommended for most setups)
celery -A worker.main worker --loglevel=info -Q audio_processing --concurrency=1

# Or use prefork with limited workers
celery -A worker.main worker --loglevel=info -Q audio_processing --concurrency=2 --pool=prefork
```

### Connecting to Remote Manager

If the Manager is running on a different machine (e.g., Docker on a Linux server):

```bash
# Point to Manager's Redis instance
export REDIS_URL=redis://your-manager-host:6379/0

celery -A worker.main worker --loglevel=info -Q audio_processing
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string (must match Manager's Redis) |

## Task States

The worker reports progress through Celery task states:

| State | Description |
|-------|-------------|
| `DOWNLOADING` | Fetching audio from source URL |
| `TRANSCRIBING` | Running Whisper transcription |
| `ANALYZING` | LLM detecting ad segments |
| `CUTTING` | Removing ads and applying crossfades |
| `UPLOADING` | Sending cleaned audio to Manager |
| `SUCCESS` | Processing complete |
| `FAILURE` | An error occurred |

## Ad Detection Logic

The LLM looks for these patterns in the transcript:

- Sponsor mentions ("This episode is sponsored by", "Thanks to our sponsor")
- Promo codes ("Use code", "promo code", "discount code")
- Call-to-action phrases ("Link in show notes", "Visit our website")
- Product pitches with URLs or special offers
- Mid-roll ad transitions ("We'll be right back", "And now a word from")

## Memory Requirements

| Component | Approximate Memory |
|-----------|-------------------|
| Whisper model | ~1.5 GB |
| LLM model | ~4.5 GB |
| Audio processing buffer | ~500 MB (varies with episode length) |
| **Total recommended** | **8+ GB unified memory** |

For Macs with 8GB RAM, use `--concurrency=1` to avoid memory pressure.

## macOS Local Network Permissions

On macOS, Python may be blocked from accessing local network resources (like a remote Redis server) due to privacy permissions. If you see "No route to host" errors, use the AppleScript workaround:

### Download Models

Get a Hugging Face token at https://huggingface.co/settings/tokens (read access is enough).

Open Script Editor and run `scripts/download_models.scpt` (replace `your_token_here` with your token), or paste:

```applescript
do shell script "cd /Users/kait/Dev/PurePod/worker && HF_TOKEN=your_token_here .venv/bin/python scripts/download_models.py"
```

### Start Worker

Open Script Editor and run `scripts/start_worker.scpt`, or paste:

```applescript
do shell script "cd /Users/kait/Dev/PurePod/worker && .venv/bin/python -m celery -A worker.main worker --loglevel=info -Q audio_processing --pool=solo >> /tmp/celery-worker.log 2>&1 &"
```

Logs are written to `/tmp/celery-worker.log`. The `--pool=solo` flag is required to avoid MLX crashes with forked processes.

### Stop Worker

```bash
pkill -f "celery -A worker.main"
```

## Troubleshooting

### "No module named 'mlx'"

MLX only works on Apple Silicon. Make sure you're running on an M1/M2/M3/M4 Mac.

### Worker not receiving tasks

1. Check Redis connectivity: `redis-cli ping`
2. Verify `REDIS_URL` matches Manager's configuration
3. Ensure the queue name matches: `-Q audio_processing`

### Out of memory errors

- Reduce concurrency: `--concurrency=1`
- Close other memory-intensive applications
- Consider a Mac with more unified memory

### ffmpeg not found

```bash
# Install ffmpeg
brew install ffmpeg

# Verify installation
ffmpeg -version
```

### Models downloading slowly

Models are cached in `~/.cache/huggingface/`. First run will download ~6GB of models. Subsequent runs use the cache.

## Development

### Running Tests

```bash
cd worker
source .venv/bin/activate

# Test transcription
python -c "from worker.main import transcribe_audio; print('Whisper OK')"

# Test LLM loading
python -c "from worker.main import get_llm; get_llm(); print('LLM OK')"
```

### Local Testing with a Sample File

```bash
python -c "
from worker.main import transcribe_audio, detect_ad_segments

# Transcribe a test file
result = transcribe_audio('test_podcast.mp3')
print(f'Transcribed {len(result[\"segments\"])} segments')

# Detect ads
ads = detect_ad_segments(result)
print(f'Found {len(ads)} ad segments')
"
```

## Network Requirements

The Worker needs network access to:

1. **Redis** (Manager's broker) - for receiving tasks and reporting status
2. **Podcast audio URLs** - to download source audio
3. **Manager API** - to upload cleaned audio (`/upload/{episode_id}`)
4. **Hugging Face Hub** (first run only) - to download models
