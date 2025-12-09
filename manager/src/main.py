import asyncio
import logging
import xml.etree.ElementTree as ET
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Set

import aiofiles
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, WebSocket, WebSocketDisconnect, APIRouter
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlmodel import Session, select

from .config import AUDIO_STORAGE_PATH, PUBLIC_HOSTNAME

STATIC_DIR = Path(__file__).parent.parent / "static"
from .database import init_db, engine
from .models import Feed, Episode, EpisodeStatus, generate_slug
from .ingest import ingest_feed, extract_feed_metadata
from .tasks import dispatch_episode_processing, sync_task_statuses

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for progress updates."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.add(connection)
        self.active_connections -= disconnected


manager = ConnectionManager()


async def task_status_sync_loop():
    """Background loop that syncs Celery task statuses to episode statuses."""
    while True:
        try:
            await asyncio.sleep(10)  # Check every 10 seconds
            with Session(engine) as session:
                result = sync_task_statuses(session)
                if result["failed"] > 0 or result["processing"] > 0:
                    logger.info(f"Task status sync: {result}")
        except Exception as e:
            logger.error(f"Error in task status sync loop: {e}")
            await asyncio.sleep(30)  # Wait longer on error


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup and start background tasks."""
    init_db()
    # Start background task sync loop
    sync_task = asyncio.create_task(task_status_sync_loop())
    yield
    # Cancel background task on shutdown
    sync_task.cancel()
    try:
        await sync_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="PodcastPurifier Manager",
    description="Manager service for PodcastPurifier - strips ads from podcasts",
    version="0.1.0",
    lifespan=lifespan,
)

# Create API router - all API endpoints go here
api = APIRouter(prefix="/api")

# Mount static files for serving cleaned audio
app.mount("/files", StaticFiles(directory=str(AUDIO_STORAGE_PATH)), name="files")


def get_db_session():
    """Dependency for database sessions."""
    with Session(engine) as session:
        yield session


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.websocket("/ws/progress")
async def websocket_progress(websocket: WebSocket):
    """
    WebSocket endpoint for real-time progress updates.

    Clients connect here to receive progress updates for episodes
    currently being processed by workers.
    """
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, wait for messages (or disconnection)
            # In production, workers would POST progress updates that get broadcast
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@api.post("/progress/{episode_id}")
async def update_progress(
    episode_id: int,
    progress: int,
    stage: str = "processing",
):
    """
    Endpoint for workers to report processing progress.

    This broadcasts the progress to all connected WebSocket clients.
    """
    await manager.broadcast({
        "episode_id": episode_id,
        "progress": progress,
        "stage": stage,
    })
    return {"status": "ok"}


@api.post("/upload/{episode_id}")
async def upload_cleaned_audio(
    episode_id: int,
    file: UploadFile = File(...),
    session: Session = Depends(get_db_session),
):
    """
    Upload a cleaned MP3 file from the Worker.

    This endpoint allows the Worker to upload the final cleaned MP3
    back to the Manager after processing. Files are stored either
    locally or in S3-compatible storage based on configuration.
    """
    from .storage import save_audio_file

    episode = session.get(Episode, episode_id)
    if not episode:
        raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")

    if not file.filename or not file.filename.endswith(".mp3"):
        raise HTTPException(status_code=400, detail="Only MP3 files are accepted")

    # Build descriptive filename: date-podcastabbrev-episode-title.mp3
    feed = session.get(Feed, episode.feed_id)

    # Get date prefix (YYYY-MM-DD)
    if episode.published_at:
        date_prefix = episode.published_at.strftime("%Y-%m-%d")
    else:
        date_prefix = datetime.utcnow().strftime("%Y-%m-%d")

    # Create podcast abbreviation from feed title (e.g., "My Cool Podcast" -> "MCP")
    if feed and feed.title:
        words = feed.title.split()
        abbrev = "".join(w[0].upper() for w in words if w and w[0].isalpha())[:4]
    else:
        abbrev = "EP"

    # Create slug from episode title
    from .models import generate_slug
    title_slug = generate_slug(episode.title)[:50]  # Limit length

    safe_filename = f"{date_prefix}-{abbrev}-{title_slug}.mp3"

    try:
        # Save to storage (local or S3)
        storage_key = save_audio_file(file.file, episode.feed_id, safe_filename)
    except Exception as e:
        logger.error(f"Failed to save file for episode {episode_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save file")

    episode.local_filename = storage_key
    episode.status = EpisodeStatus.CLEANED
    episode.error_message = None  # Clear any previous error
    episode.updated_at = datetime.utcnow()
    session.add(episode)
    session.commit()

    # Broadcast completion
    await manager.broadcast({
        "episode_id": episode_id,
        "progress": 100,
        "stage": "completed",
    })

    logger.info(f"Uploaded cleaned audio for episode {episode_id}: {storage_key}")

    return {
        "message": "File uploaded successfully",
        "episode_id": episode_id,
        "local_filename": storage_key,
    }


def make_unique_slug(session: Session, base_slug: str) -> str:
    """Generate a unique slug, appending a number if necessary."""
    slug = base_slug
    counter = 2
    while session.exec(select(Feed).where(Feed.slug == slug)).first():
        slug = f"{base_slug}-{counter}"
        counter += 1
    return slug


@api.post("/feeds", response_model=Feed)
async def create_feed(
    rss_url: str,
    session: Session = Depends(get_db_session),
):
    """Create a new feed by fetching metadata from the RSS URL."""
    existing = session.exec(select(Feed).where(Feed.rss_url == rss_url)).first()
    if existing:
        raise HTTPException(status_code=400, detail="Feed with this URL already exists")

    # Fetch metadata from the RSS feed
    metadata = extract_feed_metadata(rss_url)
    if not metadata:
        raise HTTPException(status_code=400, detail="Could not parse RSS feed. Please check the URL.")

    # Append "(Purified)" to the title
    title = f"{metadata['title']} (Purified)"

    # Generate a unique slug from the title
    base_slug = generate_slug(title)
    slug = make_unique_slug(session, base_slug)

    feed = Feed(
        slug=slug,
        title=title,
        rss_url=rss_url,
        description=metadata.get('description'),
        image_url=metadata.get('image_url'),
        author=metadata.get('author'),
    )
    session.add(feed)
    session.commit()
    session.refresh(feed)

    # Auto-ingest episodes from the feed
    try:
        new_episodes = ingest_feed(feed.id, session)
        logger.info(f"Auto-ingested {len(new_episodes)} episodes for new feed {feed.id}")
    except Exception as e:
        logger.error(f"Failed to auto-ingest episodes for feed {feed.id}: {e}")

    return feed


@api.get("/feeds", response_model=List[Feed])
async def list_feeds(session: Session = Depends(get_db_session)):
    """List all feeds."""
    feeds = session.exec(select(Feed)).all()
    return feeds


@api.patch("/feeds/{feed_id}/auto-process", response_model=Feed)
async def update_feed_auto_process(
    feed_id: int,
    auto_process: bool,
    session: Session = Depends(get_db_session),
):
    """Update the auto_process setting for a feed."""
    feed = session.get(Feed, feed_id)
    if not feed:
        raise HTTPException(status_code=404, detail=f"Feed {feed_id} not found")

    feed.auto_process = auto_process
    feed.updated_at = datetime.utcnow()
    session.add(feed)
    session.commit()
    session.refresh(feed)

    return feed


@api.delete("/feeds/{feed_id}")
async def delete_feed(
    feed_id: int,
    session: Session = Depends(get_db_session),
):
    """Delete a feed and all its episodes."""
    feed = session.get(Feed, feed_id)
    if not feed:
        raise HTTPException(status_code=404, detail=f"Feed {feed_id} not found")

    # Delete all episodes for this feed
    from sqlalchemy import delete
    result = session.execute(delete(Episode).where(Episode.feed_id == feed_id))
    episode_count = result.rowcount

    # Delete the feed
    session.delete(feed)
    session.commit()

    logger.info(f"Deleted feed {feed_id} and {episode_count} episodes")

    return {"message": f"Feed {feed_id} deleted", "deleted_episodes": episode_count}


@api.post("/feeds/{feed_id}/ingest")
async def trigger_ingest(feed_id: int, session: Session = Depends(get_db_session)):
    """Trigger ingestion for a specific feed."""
    feed = session.get(Feed, feed_id)
    if not feed:
        raise HTTPException(status_code=404, detail=f"Feed {feed_id} not found")

    new_episodes = ingest_feed(feed_id, session)

    return {
        "message": f"Ingestion complete for feed {feed_id}",
        "new_episodes": len(new_episodes),
        "episodes": [{"id": ep.id, "title": ep.title, "status": ep.status} for ep in new_episodes],
    }


ITUNES_NS = "http://www.itunes.com/dtds/podcast-1.0.dtd"


def format_duration(seconds: int) -> str:
    """Format duration as HH:MM:SS or MM:SS."""
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


@app.get("/feed/{slug}")
async def get_feed_rss(slug: str, session: Session = Depends(get_db_session)):
    """
    Generate RSS 2.0 XML feed for cleaned episodes.

    Returns an RSS feed with enclosure tags pointing to the public URL
    for each cleaned episode's audio file. Includes iTunes podcast metadata.
    """
    feed = session.exec(select(Feed).where(Feed.slug == slug)).first()
    if not feed:
        raise HTTPException(status_code=404, detail=f"Feed '{slug}' not found")

    # Get only cleaned episodes for this feed, ordered by publish date
    episodes = session.exec(
        select(Episode)
        .where(Episode.feed_id == feed.id)
        .where(Episode.status == EpisodeStatus.CLEANED)
        .order_by(Episode.published_at.desc().nullslast())
    ).all()

    # Build RSS 2.0 XML with iTunes namespace
    rss = ET.Element("rss", version="2.0")
    rss.set("xmlns:itunes", ITUNES_NS)
    channel = ET.SubElement(rss, "channel")

    # Channel metadata
    ET.SubElement(channel, "title").text = feed.title
    ET.SubElement(channel, "link").text = f"{PUBLIC_HOSTNAME}/feed/{feed.slug}"
    ET.SubElement(channel, "description").text = feed.description or f"Ad-free version of {feed.title}"
    ET.SubElement(channel, "lastBuildDate").text = datetime.utcnow().strftime(
        "%a, %d %b %Y %H:%M:%S +0000"
    )
    ET.SubElement(channel, "generator").text = "PurePod"

    # iTunes channel metadata
    if feed.author:
        ET.SubElement(channel, f"{{{ITUNES_NS}}}author").text = feed.author
    if feed.image_url:
        ET.SubElement(channel, f"{{{ITUNES_NS}}}image", href=feed.image_url)
        # Also add standard RSS image
        image = ET.SubElement(channel, "image")
        ET.SubElement(image, "url").text = feed.image_url
        ET.SubElement(image, "title").text = feed.title
        ET.SubElement(image, "link").text = f"{PUBLIC_HOSTNAME}/feed/{feed.slug}"

    for episode in episodes:
        if not episode.local_filename:
            continue

        item = ET.SubElement(channel, "item")
        ET.SubElement(item, "title").text = episode.title
        ET.SubElement(item, "guid").text = episode.guid

        # Use original publish date if available
        if episode.published_at:
            pub_date = episode.published_at.strftime("%a, %d %b %Y %H:%M:%S +0000")
        else:
            pub_date = episode.updated_at.strftime("%a, %d %b %Y %H:%M:%S +0000")
        ET.SubElement(item, "pubDate").text = pub_date

        # Episode description/show notes
        if episode.description:
            ET.SubElement(item, "description").text = episode.description
            ET.SubElement(item, f"{{{ITUNES_NS}}}summary").text = episode.description

        # Episode duration
        if episode.duration:
            ET.SubElement(item, f"{{{ITUNES_NS}}}duration").text = format_duration(episode.duration)

        # Episode artwork (falls back to feed artwork)
        ep_image = episode.image_url or feed.image_url
        if ep_image:
            ET.SubElement(item, f"{{{ITUNES_NS}}}image", href=ep_image)

        # Build public URL for the audio file and get file size
        from .storage import get_audio_url, get_file_size
        audio_url = get_audio_url(episode.local_filename)
        file_size = get_file_size(episode.local_filename)

        ET.SubElement(
            item,
            "enclosure",
            url=audio_url,
            type="audio/mpeg",
            length=str(file_size),
        )

    xml_str = ET.tostring(rss, encoding="unicode", xml_declaration=True)
    return Response(content=xml_str, media_type="application/rss+xml")


class EpisodeWithFeed(BaseModel):
    """Episode with feed title for display."""
    id: int
    feed_id: int
    feed_title: str
    guid: str
    status: EpisodeStatus
    title: str
    audio_url: str
    published_at: datetime | None
    local_filename: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class PaginatedEpisodes(BaseModel):
    """Paginated episode response."""
    items: List[EpisodeWithFeed]
    total: int
    page: int
    page_size: int
    total_pages: int


@api.get("/episodes", response_model=PaginatedEpisodes)
async def list_episodes(
    feed_id: int = None,
    status: EpisodeStatus = None,
    exclude_statuses: str = None,
    page: int = 1,
    page_size: int = 25,
    session: Session = Depends(get_db_session),
):
    """List episodes with optional filters and pagination."""
    query = select(Episode)

    if feed_id is not None:
        query = query.where(Episode.feed_id == feed_id)

    if status is not None:
        query = query.where(Episode.status == status)
    elif exclude_statuses:
        # Exclude multiple statuses (comma-separated)
        excluded = [EpisodeStatus(s.strip()) for s in exclude_statuses.split(",")]
        for exc_status in excluded:
            query = query.where(Episode.status != exc_status)

    # Order by published date (newest first), fallback to created_at
    query = query.order_by(Episode.published_at.desc().nullslast(), Episode.created_at.desc())

    # Get total count
    count_query = select(Episode)
    if feed_id is not None:
        count_query = count_query.where(Episode.feed_id == feed_id)
    if status is not None:
        count_query = count_query.where(Episode.status == status)
    elif exclude_statuses:
        excluded = [EpisodeStatus(s.strip()) for s in exclude_statuses.split(",")]
        for exc_status in excluded:
            count_query = count_query.where(Episode.status != exc_status)

    all_episodes = session.exec(count_query).all()
    total = len(all_episodes)

    # Paginate
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)
    episodes = session.exec(query).all()

    # Get feed titles
    feed_ids = {ep.feed_id for ep in episodes}
    if feed_ids:
        feeds = {f.id: f for f in session.exec(select(Feed).where(Feed.id.in_(feed_ids))).all()}
    else:
        feeds = {}

    items = [
        EpisodeWithFeed(
            id=ep.id,
            feed_id=ep.feed_id,
            feed_title=feeds.get(ep.feed_id, Feed(slug="unknown", title="Unknown")).title,
            guid=ep.guid,
            status=ep.status,
            title=ep.title,
            audio_url=ep.audio_url,
            published_at=ep.published_at,
            local_filename=ep.local_filename,
            error_message=ep.error_message,
            created_at=ep.created_at,
            updated_at=ep.updated_at,
        )
        for ep in episodes
    ]

    total_pages = (total + page_size - 1) // page_size

    return PaginatedEpisodes(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


class EpisodeStats(BaseModel):
    """Episode counts by status."""
    discovered: int
    queued: int
    processing: int
    cleaned: int
    failed: int
    ignored: int
    total: int


@api.get("/episodes/stats", response_model=EpisodeStats)
async def get_episode_stats(session: Session = Depends(get_db_session)):
    """Get episode counts by status."""
    from sqlalchemy import func

    stats = {}
    for status in EpisodeStatus:
        count = session.exec(
            select(func.count()).where(Episode.status == status)
        ).one()
        stats[status.value] = count

    stats['total'] = sum(stats.values())
    return EpisodeStats(**stats)


class BulkEpisodeRequest(BaseModel):
    episode_ids: List[int]


@api.post("/episodes/ignore")
async def ignore_episodes(
    request: BulkEpisodeRequest,
    session: Session = Depends(get_db_session),
):
    """Mark episodes as ignored."""
    ignored_count = 0

    for episode_id in request.episode_ids:
        episode = session.get(Episode, episode_id)
        if episode and episode.status in (EpisodeStatus.DISCOVERED, EpisodeStatus.FAILED):
            episode.status = EpisodeStatus.IGNORED
            episode.updated_at = datetime.utcnow()
            session.add(episode)
            ignored_count += 1

    session.commit()
    return {"ignored": ignored_count}


@api.post("/episodes/unignore")
async def unignore_episodes(
    request: BulkEpisodeRequest,
    session: Session = Depends(get_db_session),
):
    """Restore ignored episodes to discovered status."""
    restored_count = 0

    for episode_id in request.episode_ids:
        episode = session.get(Episode, episode_id)
        if episode and episode.status == EpisodeStatus.IGNORED:
            episode.status = EpisodeStatus.DISCOVERED
            episode.updated_at = datetime.utcnow()
            session.add(episode)
            restored_count += 1

    session.commit()
    return {"restored": restored_count}


@api.post("/episodes/queue")
async def queue_episodes(
    request: BulkEpisodeRequest,
    session: Session = Depends(get_db_session),
):
    """
    Queue multiple episodes for processing.

    Episodes with DISCOVERED or FAILED status can be queued.
    This dispatches Celery tasks to the Worker for each episode.
    """
    queued_count = 0
    dispatched_tasks = []

    for episode_id in request.episode_ids:
        episode = session.get(Episode, episode_id)
        if episode and episode.status in (EpisodeStatus.DISCOVERED, EpisodeStatus.FAILED):
            episode.status = EpisodeStatus.QUEUED
            episode.updated_at = datetime.utcnow()
            session.add(episode)
            queued_count += 1

            # Dispatch to Worker
            try:
                task_id = dispatch_episode_processing(episode_id, episode.audio_url)
                dispatched_tasks.append({"episode_id": episode_id, "task_id": task_id})
            except Exception as e:
                logger.error(f"Failed to dispatch episode {episode_id}: {e}")

    session.commit()

    return {"queued": queued_count, "tasks": dispatched_tasks}


@api.post("/episodes/unqueue")
async def unqueue_episodes(
    request: BulkEpisodeRequest,
    session: Session = Depends(get_db_session),
):
    """
    Cancel queued episodes by moving them back to DISCOVERED.

    Note: This doesn't revoke the Celery task - if the worker picks it up,
    it will still process. This just updates the status in the database.
    """
    unqueued_count = 0

    for episode_id in request.episode_ids:
        episode = session.get(Episode, episode_id)
        if episode and episode.status == EpisodeStatus.QUEUED:
            episode.status = EpisodeStatus.DISCOVERED
            episode.updated_at = datetime.utcnow()
            session.add(episode)
            unqueued_count += 1

    session.commit()
    return {"unqueued": unqueued_count}


@api.post("/episodes/fail")
async def fail_episodes(
    request: BulkEpisodeRequest,
    session: Session = Depends(get_db_session),
):
    """
    Mark episodes as FAILED.

    Useful for manually marking stuck QUEUED or PROCESSING episodes as failed.
    """
    failed_count = 0

    for episode_id in request.episode_ids:
        episode = session.get(Episode, episode_id)
        if episode and episode.status in (EpisodeStatus.QUEUED, EpisodeStatus.PROCESSING):
            episode.status = EpisodeStatus.FAILED
            episode.updated_at = datetime.utcnow()
            session.add(episode)
            failed_count += 1

    session.commit()
    return {"failed": failed_count}


class EpisodeStatusUpdate(BaseModel):
    status: str  # processing, failed, etc.
    stage: str | None = None  # downloading, transcribing, analyzing, cutting, uploading
    error_message: str | None = None


@api.post("/episodes/{episode_id}/status")
async def update_episode_status(
    episode_id: int,
    update: EpisodeStatusUpdate,
    session: Session = Depends(get_db_session),
):
    """
    Update episode status from the worker.

    Called by the worker to report progress/state changes.
    """
    episode = session.get(Episode, episode_id)
    if not episode:
        raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")

    if update.status == "processing":
        episode.status = EpisodeStatus.PROCESSING
        episode.error_message = None  # Clear any previous error
    elif update.status == "failed":
        episode.status = EpisodeStatus.FAILED
        if update.error_message:
            episode.error_message = update.error_message

    episode.updated_at = datetime.utcnow()
    session.add(episode)
    session.commit()

    # Broadcast to WebSocket clients
    await manager.broadcast({
        "episode_id": episode_id,
        "status": update.status,
        "stage": update.stage,
    })

    return {"ok": True}


# Include API router
app.include_router(api)

# Serve frontend static files (must be after API routes)
if STATIC_DIR.exists():
    @app.get("/")
    async def serve_spa_root():
        """Serve the SPA index.html at root."""
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/{path:path}")
    async def serve_spa_fallback(path: str):
        """Serve static files or index.html for SPA client-side routing."""
        file_path = STATIC_DIR / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        # Return index.html for SPA routes
        return FileResponse(STATIC_DIR / "index.html")
