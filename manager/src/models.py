import enum
import re
from typing import Optional, List
from datetime import datetime

from sqlmodel import SQLModel, Field, Relationship


def generate_slug(title: str) -> str:
    """Generate a URL-friendly slug from a title."""
    # Remove "(Purified)" suffix if present for cleaner slugs
    title = re.sub(r'\s*\(Purified\)\s*$', '', title, flags=re.IGNORECASE)
    # Convert to lowercase and replace non-alphanumeric chars with hyphens
    slug = re.sub(r'[^a-z0-9]+', '-', title.lower())
    # Remove leading/trailing hyphens
    slug = slug.strip('-')
    return slug or 'feed'


class EpisodeStatus(str, enum.Enum):
    DISCOVERED = "discovered"
    QUEUED = "queued"
    PROCESSING = "processing"
    CLEANED = "cleaned"
    FAILED = "failed"
    IGNORED = "ignored"


class Feed(SQLModel, table=True):
    __tablename__ = "feeds"

    id: Optional[int] = Field(default=None, primary_key=True)
    slug: str = Field(unique=True, index=True)
    title: str = Field(index=True)
    rss_url: str = Field(unique=True)
    description: Optional[str] = Field(default=None)
    image_url: Optional[str] = Field(default=None)
    author: Optional[str] = Field(default=None)
    auto_process: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    episodes: List["Episode"] = Relationship(back_populates="feed")


class Episode(SQLModel, table=True):
    __tablename__ = "episodes"

    id: Optional[int] = Field(default=None, primary_key=True)
    feed_id: int = Field(foreign_key="feeds.id", index=True)
    guid: str = Field(index=True)
    status: EpisodeStatus = Field(default=EpisodeStatus.DISCOVERED)
    title: str
    audio_url: str
    description: Optional[str] = Field(default=None)
    duration: Optional[int] = Field(default=None)  # Duration in seconds
    image_url: Optional[str] = Field(default=None)
    published_at: Optional[datetime] = Field(default=None)
    local_filename: Optional[str] = Field(default=None)
    error_message: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    feed: Optional[Feed] = Relationship(back_populates="episodes")

    class Config:
        use_enum_values = True
