"""
Celery task dispatching for the Manager.

This module provides functions to dispatch tasks to the remote Worker.
"""
import os
import logging
from datetime import datetime

from celery import Celery
from celery.result import AsyncResult
from sqlmodel import Session, select

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
MANAGER_BASE_URL = os.getenv("MANAGER_BASE_URL", "http://web:8000")

# Create Celery app for dispatching tasks to Worker
celery_app = Celery(
    "podcastpurifier_manager",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)


def dispatch_episode_processing(episode_id: int, audio_url: str) -> str:
    """
    Dispatch an episode for processing by the Worker.

    Args:
        episode_id: The episode ID in the database.
        audio_url: URL to download the original audio.

    Returns:
        The Celery task ID.
    """
    callback_url = f"{MANAGER_BASE_URL}/api/upload/{episode_id}"

    task = celery_app.send_task(
        "worker.process_episode",
        args=[episode_id, audio_url, callback_url],
        queue="audio_processing",
    )

    logger.info(f"Dispatched episode {episode_id} for processing, task_id={task.id}")
    return task.id


def sync_task_statuses(session: Session) -> dict:
    """
    Check Celery task results and update episode statuses accordingly.

    Looks for episodes in QUEUED or PROCESSING state and checks if their
    Celery tasks have completed (success or failure).

    Returns:
        Dict with counts of updated episodes.
    """
    from .models import Episode, EpisodeStatus

    # Find episodes that are queued or processing
    episodes = session.exec(
        select(Episode).where(
            Episode.status.in_([EpisodeStatus.QUEUED, EpisodeStatus.PROCESSING])
        )
    ).all()

    if not episodes:
        return {"checked": 0, "failed": 0, "processing": 0}

    failed_count = 0
    processing_count = 0

    # Check each episode's task status via Redis
    # Task results are stored with key pattern: celery-task-meta-<task_id>
    # But we don't store task_id on episodes, so we check by episode_id pattern

    import redis
    r = redis.from_url(REDIS_URL)

    for episode in episodes:
        # Look for task results that contain this episode_id
        # Celery stores results as: celery-task-meta-{task_id}
        # We need to scan for results containing our episode_id

        # Search for task metadata keys
        for key in r.scan_iter("celery-task-meta-*"):
            try:
                result_data = r.get(key)
                if not result_data:
                    continue

                import json
                result = json.loads(result_data)

                # Check if this result is for our episode
                task_result = result.get("result", {})
                if not isinstance(task_result, dict):
                    continue

                result_episode_id = task_result.get("episode_id")
                if result_episode_id != episode.id:
                    continue

                # Found a matching result
                status = result.get("status")

                if status == "FAILURE" or task_result.get("status") == "error":
                    episode.status = EpisodeStatus.FAILED
                    episode.updated_at = datetime.utcnow()
                    session.add(episode)
                    failed_count += 1
                    logger.info(f"Marked episode {episode.id} as FAILED based on task result")
                    # Delete the processed result key
                    r.delete(key)
                    break
                elif status == "SUCCESS" and task_result.get("status") == "success":
                    # Episode should already be CLEANED from upload callback
                    # But if not, we could handle it here
                    r.delete(key)
                    break
                elif status in ("PENDING", "STARTED", "RETRY"):
                    # Task is still running, update to PROCESSING if needed
                    if episode.status == EpisodeStatus.QUEUED:
                        episode.status = EpisodeStatus.PROCESSING
                        episode.updated_at = datetime.utcnow()
                        session.add(episode)
                        processing_count += 1
                    break

            except Exception as e:
                logger.debug(f"Error checking task result {key}: {e}")
                continue

    session.commit()

    return {
        "checked": len(episodes),
        "failed": failed_count,
        "processing": processing_count,
    }
