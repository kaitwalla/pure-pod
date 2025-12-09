# PodcastPurifier Manager

The Manager is the central orchestration service for PodcastPurifier. It handles RSS feed ingestion, database management, the web UI, and serves the final ad-free podcast feeds.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Manager                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  FastAPI    │  │  PostgreSQL │  │  Redis              │  │
│  │  (web)      │◄─┤  (db)       │  │  (message broker)   │  │
│  └──────┬──────┘  └─────────────┘  └──────────┬──────────┘  │
│         │                                      │             │
│         │ Celery Tasks                         │             │
│         └──────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Remote Worker  │
                    │  (ML Processing)│
                    └─────────────────┘
```

## Components

### Backend (`src/`)

| File | Purpose |
|------|---------|
| `main.py` | FastAPI application with all REST endpoints and WebSocket support |
| `models.py` | SQLModel database models (Feed, Episode) |
| `database.py` | Database connection and initialization |
| `ingest.py` | RSS feed parsing and episode discovery |
| `tasks.py` | Celery task dispatching to remote Worker |
| `config.py` | Environment variable configuration |

### Frontend (`frontend/`)

React + TypeScript SPA built with Vite, TailwindCSS, and shadcn/ui components.

- **FeedView**: Displays subscribed podcasts with auto-process toggle
- **EpisodeTable**: Episode inbox with bulk selection and queue actions

## API Endpoints

### Feeds

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/feeds` | List all feeds |
| `POST` | `/feeds?title=...&rss_url=...` | Create a new feed |
| `PATCH` | `/feeds/{id}/auto-process?auto_process=true` | Toggle auto-processing |
| `POST` | `/feeds/{id}/ingest` | Trigger RSS ingestion |
| `GET` | `/feed/{id}` | Get RSS 2.0 XML for cleaned episodes |

### Episodes

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/episodes?feed_id=...&status=...` | List episodes (with optional filters) |
| `POST` | `/episodes/queue` | Queue episodes for processing (body: `{"episode_ids": [...]}`) |

### Worker Callbacks

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload/{episode_id}` | Worker uploads cleaned MP3 |
| `POST` | `/progress/{episode_id}?progress=...&stage=...` | Worker reports progress |

### Other

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/files/{path}` | Serve cleaned audio files |
| `WS` | `/ws/progress` | WebSocket for real-time progress updates |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://podcast:podcast@localhost:5432/podcastpurifier` | PostgreSQL connection string |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection for Celery |
| `AUDIO_STORAGE_PATH` | `/app/audio` | Directory for cleaned audio files |
| `PUBLIC_HOSTNAME` | `localhost` | Public hostname for RSS feed URLs (e.g., your Tailscale Funnel URL) |
| `MANAGER_BASE_URL` | `http://web:8000` | Internal URL for Worker callbacks |

## Running with Docker Compose

From the project root:

```bash
# Start all services (Manager + PostgreSQL + Redis)
docker compose up -d

# View logs
docker compose logs -f web

# Stop services
docker compose down
```

### With Tailscale Funnel

To expose the Manager publicly via Tailscale Funnel:

```bash
# Set your Tailscale Funnel hostname
export PUBLIC_HOSTNAME=your-machine.tail1234.ts.net

# Start services
docker compose up -d

# Enable Tailscale Funnel (run on host, not in container)
tailscale funnel 8000
```

The RSS feed URLs in `/feed/{id}` will use `https://{PUBLIC_HOSTNAME}/files/...` for enclosure URLs.

## Local Development

### Backend

```bash
cd manager

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run with uvicorn (requires PostgreSQL and Redis running)
DATABASE_URL=postgresql://podcast:podcast@localhost:5432/podcastpurifier \
REDIS_URL=redis://localhost:6379/0 \
uvicorn src.main:app --reload
```

### Frontend

```bash
cd manager/frontend

# Install dependencies
npm install

# Run dev server (proxies API to localhost:8000)
npm run dev

# Build for production
npm run build
```

## Episode Status Flow

```
DISCOVERED ──► QUEUED ──► PROCESSING ──► CLEANED
                              │
                              └──► FAILED
```

1. **DISCOVERED**: Episode found during RSS ingestion
2. **QUEUED**: User selected episode for processing (or auto-process enabled)
3. **PROCESSING**: Worker is actively processing the episode
4. **CLEANED**: Processing complete, cleaned audio available
5. **FAILED**: Processing failed (check Worker logs)

## Auto-Process Feature

When `auto_process` is enabled for a feed:
- **New** episodes discovered during ingestion are automatically set to `QUEUED`
- Existing episodes are NOT affected (won't be retroactively queued)
- Celery tasks are dispatched to the Worker for each queued episode
