"""
Shared Celery task definitions.

These are task signatures that can be used by the Manager to queue work.
The actual implementation lives on the Worker node.
"""
from celery import Celery
from . import celery_config

# Create Celery app with shared config
celery_app = Celery("podcastpurifier")
celery_app.config_from_object(celery_config)


@celery_app.task(name="shared.tasks.process_episode", bind=True)
def process_episode(self, episode_id: int, audio_url: str, manager_upload_url: str):
    """
    Task to process an episode's audio and strip ads.

    This is a stub - the actual implementation is on the Worker.

    Args:
        episode_id: The episode ID in the database.
        audio_url: URL to download the original audio.
        manager_upload_url: URL to POST the cleaned audio back to Manager.
    """
    raise NotImplementedError("This task runs on the Worker node")
