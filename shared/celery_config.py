import os

# Celery configuration shared between Manager and Worker

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Broker settings
broker_url = REDIS_URL
result_backend = REDIS_URL

# Task settings
task_serializer = "json"
result_serializer = "json"
accept_content = ["json"]
timezone = "UTC"
enable_utc = True

# Task routing - Worker processes these tasks
task_routes = {
    "shared.tasks.process_episode": {"queue": "audio_processing"},
}
