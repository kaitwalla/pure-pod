"""
Storage abstraction for audio files.
Supports local filesystem and S3-compatible storage (DigitalOcean Spaces, AWS S3, etc.)
"""
import logging
from pathlib import Path
from typing import BinaryIO

import boto3
from botocore.config import Config

from .config import (
    S3_ENABLED,
    S3_ENDPOINT_URL,
    S3_BUCKET,
    S3_ACCESS_KEY,
    S3_SECRET_KEY,
    S3_REGION,
    S3_PUBLIC_URL,
    AUDIO_STORAGE_PATH,
    PUBLIC_HOSTNAME,
)

logger = logging.getLogger(__name__)

_s3_client = None


def get_s3_client():
    """Get or create S3 client (lazy initialization)."""
    global _s3_client
    if _s3_client is None and S3_ENABLED:
        _s3_client = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT_URL,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
            region_name=S3_REGION,
            config=Config(signature_version="s3v4"),
        )
    return _s3_client


def save_audio_file(file_content: BinaryIO, feed_id: int, filename: str) -> str:
    """
    Save an audio file to storage.

    Args:
        file_content: File-like object with audio content
        feed_id: Feed ID for organizing files
        filename: Original filename

    Returns:
        Storage path/key for the file (relative path for local, S3 key for S3)
    """
    # Create a safe filename with feed_id prefix
    safe_filename = filename.replace("/", "_")
    storage_key = f"{feed_id}/{safe_filename}"

    if S3_ENABLED:
        return _save_to_s3(file_content, storage_key)
    else:
        return _save_to_local(file_content, storage_key)


def _save_to_s3(file_content: BinaryIO, storage_key: str) -> str:
    """Save file to S3-compatible storage."""
    client = get_s3_client()

    client.upload_fileobj(
        file_content,
        S3_BUCKET,
        storage_key,
        ExtraArgs={
            "ContentType": "audio/mpeg",
            "ACL": "public-read",
        },
    )

    logger.info(f"Uploaded to S3: {storage_key}")
    return storage_key


def _save_to_local(file_content: BinaryIO, storage_key: str) -> str:
    """Save file to local filesystem."""
    file_path = AUDIO_STORAGE_PATH / storage_key
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "wb") as f:
        # Read in chunks to handle large files
        while chunk := file_content.read(8192):
            f.write(chunk)

    logger.info(f"Saved locally: {file_path}")
    return storage_key


def get_audio_url(storage_key: str) -> str:
    """
    Get the public URL for an audio file.

    Args:
        storage_key: Storage path/key returned from save_audio_file

    Returns:
        Public URL for the audio file
    """
    if S3_ENABLED:
        # Use CDN URL if configured, otherwise construct from endpoint
        if S3_PUBLIC_URL:
            # Include bucket name in path for CDN URLs
            return f"{S3_PUBLIC_URL.rstrip('/')}/{S3_BUCKET}/{storage_key}"
        else:
            # Construct URL from endpoint (works for DigitalOcean Spaces)
            # e.g., https://bucket.nyc3.digitaloceanspaces.com/key
            endpoint = S3_ENDPOINT_URL.replace("https://", "")
            return f"https://{S3_BUCKET}.{endpoint}/{storage_key}"
    else:
        # Local storage - serve via /files/ endpoint
        return f"{PUBLIC_HOSTNAME}/files/{storage_key}"


def get_file_size(storage_key: str) -> int:
    """
    Get the size of a stored file in bytes.

    Args:
        storage_key: Storage path/key

    Returns:
        File size in bytes, or 0 if not found
    """
    if S3_ENABLED:
        try:
            client = get_s3_client()
            response = client.head_object(Bucket=S3_BUCKET, Key=storage_key)
            return response.get("ContentLength", 0)
        except Exception as e:
            logger.warning(f"Failed to get S3 file size for {storage_key}: {e}")
            return 0
    else:
        try:
            file_path = AUDIO_STORAGE_PATH / storage_key
            return file_path.stat().st_size
        except OSError:
            return 0


def delete_audio_file(storage_key: str) -> bool:
    """
    Delete an audio file from storage.

    Args:
        storage_key: Storage path/key

    Returns:
        True if deleted, False if failed
    """
    if S3_ENABLED:
        try:
            client = get_s3_client()
            client.delete_object(Bucket=S3_BUCKET, Key=storage_key)
            logger.info(f"Deleted from S3: {storage_key}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete S3 file {storage_key}: {e}")
            return False
    else:
        try:
            file_path = AUDIO_STORAGE_PATH / storage_key
            file_path.unlink(missing_ok=True)
            logger.info(f"Deleted locally: {file_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete local file {storage_key}: {e}")
            return False
