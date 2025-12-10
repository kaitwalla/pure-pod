import os
from pathlib import Path

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://podcast:podcast@localhost:5432/podcastpurifier")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
AUDIO_STORAGE_PATH = Path(os.getenv("AUDIO_STORAGE_PATH", "/app/audio"))
PUBLIC_HOSTNAME = os.getenv("PUBLIC_HOSTNAME", "localhost")

# S3-compatible storage configuration
# Set S3_ENABLED=true to use S3 instead of local storage
S3_ENABLED = os.getenv("S3_ENABLED", "false").lower() == "true"
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "")  # e.g., https://nyc3.digitaloceanspaces.com
S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "")
S3_REGION = os.getenv("S3_REGION", "nyc3")
S3_PUBLIC_URL = os.getenv("S3_PUBLIC_URL", "")  # e.g., https://bucket.nyc3.cdn.digitaloceanspaces.com

# Worker authentication - required for upload endpoint
WORKER_API_KEY = os.getenv("WORKER_API_KEY", "")

# Ensure audio storage directory exists (for local storage or temp files)
AUDIO_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
