# PurePod

A self-hosted podcast ad removal system that automatically transcribes, detects, and removes advertisements from podcast episodes using local ML models.

## How It Works

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Podcast Feed   │────▶│     Manager     │────▶│  Clean Feed     │
│  (with ads)     │     │  (orchestrator) │     │  (ad-free)      │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                                 │ Celery/Redis
                                 ▼
                        ┌─────────────────┐
                        │     Worker      │
                        │ (Apple Silicon) │
                        │                 │
                        │ • Whisper       │
                        │ • Llama 3       │
                        └─────────────────┘
```

1. **Subscribe** to a podcast via its RSS feed
2. **Queue** episodes for processing (or enable auto-process)
3. **Worker** downloads, transcribes, detects ads, and removes them
4. **Subscribe** to the clean feed URL in your podcast app

## Components

| Component | Description | Runs On |
|-----------|-------------|---------|
| **Manager** | Web UI, API, RSS feed generation, task coordination | Docker (any platform) |
| **Worker** | ML processing (transcription + ad detection) | Apple Silicon Mac |

The Manager runs in Docker with PostgreSQL and Redis. The Worker runs natively on macOS to leverage MLX for fast on-device ML inference.

## Quick Start

### 1. Start the Manager

```bash
# Clone and configure
git clone https://github.com/your-repo/purepod.git
cd purepod

# Set environment variables
export PUBLIC_HOSTNAME=https://your-public-url.com
export S3_ENABLED=true
export S3_ENDPOINT_URL=https://nyc3.digitaloceanspaces.com
export S3_BUCKET=your-bucket
export S3_ACCESS_KEY=your-key
export S3_SECRET_KEY=your-secret
export S3_PUBLIC_URL=https://your-bucket.nyc3.cdn.digitaloceanspaces.com

# Start services
docker compose up -d
```

The web UI will be available at `http://localhost:8000`.

### 2. Start the Worker (on Apple Silicon Mac)

```bash
cd worker

# Install dependencies
brew install ffmpeg
uv sync

# Configure Redis connection to Manager
export REDIS_URL=redis://your-manager-host:6379/0

# Start worker
source .venv/bin/activate
celery -A worker.main worker --loglevel=info -Q audio_processing --pool=solo
```

### 3. Add a Podcast

1. Open the web UI at `http://localhost:8000`
2. Add a podcast RSS feed URL
3. Queue episodes for processing (or enable auto-process)
4. Once processed, subscribe to your clean feed: `https://your-public-url.com/feed/{slug}`

## Storage Options

### Local Storage (default)
Cleaned audio files are stored in a Docker volume and served via the Manager API.

### S3-Compatible Storage
For production use with DigitalOcean Spaces, AWS S3, or similar:

| Variable | Description |
|----------|-------------|
| `S3_ENABLED` | Set to `true` to enable |
| `S3_ENDPOINT_URL` | e.g., `https://nyc3.digitaloceanspaces.com` |
| `S3_BUCKET` | Bucket name |
| `S3_ACCESS_KEY` | Access key |
| `S3_SECRET_KEY` | Secret key |
| `S3_REGION` | Region (default: `nyc3`) |
| `S3_PUBLIC_URL` | CDN URL for public access |

## ML Models

| Model | Purpose | Size |
|-------|---------|------|
| distil-whisper-large-v3 | Audio transcription with word timestamps | ~1.5GB |
| Llama-3-8B-Instruct-4bit | Ad segment detection from transcript | ~5.3GB |

Models are downloaded automatically on first use and cached in `~/.cache/huggingface/`.

## Requirements

### Manager
- Docker and Docker Compose
- Public URL for RSS feed access (Tailscale Funnel, Cloudflare Tunnel, etc.)

### Worker
- Apple Silicon Mac (M1/M2/M3/M4)
- 8GB+ unified memory recommended
- Python 3.11+
- ffmpeg

## Documentation

- [Manager README](manager/README.md) - API endpoints, configuration, development
- [Worker README](worker/README.md) - ML pipeline, troubleshooting, memory requirements

## Episode Filename Format

Cleaned episodes are stored with descriptive filenames:
```
2024-01-15-MCP-episode-title-slug.mp3
```
- Date from episode publish date
- Podcast abbreviation (initials of feed title)
- Episode title as URL-friendly slug
