import logging
from typing import List, Optional
from datetime import datetime
from email.utils import parsedate_to_datetime

import feedparser
from sqlmodel import Session, select

from .models import Feed, Episode, EpisodeStatus
from .database import get_session

logger = logging.getLogger(__name__)


def parse_rss(rss_url: str) -> Optional[feedparser.FeedParserDict]:
    """
    Parse an RSS feed URL and return the parsed feed.

    Args:
        rss_url: The URL of the RSS feed to parse.

    Returns:
        Parsed feed object or None if parsing failed.
    """
    try:
        feed = feedparser.parse(rss_url)
        if feed.bozo and not feed.entries:
            logger.error(f"Failed to parse RSS feed {rss_url}: {feed.bozo_exception}")
            return None
        return feed
    except Exception as e:
        logger.error(f"Error parsing RSS feed {rss_url}: {e}")
        return None


def extract_feed_metadata(rss_url: str) -> Optional[dict]:
    """
    Extract metadata from an RSS feed.

    Args:
        rss_url: The URL of the RSS feed.

    Returns:
        Dictionary with title, description, image_url, author or None if parsing failed.
    """
    parsed = parse_rss(rss_url)
    if not parsed:
        return None

    feed_info = parsed.feed

    # Extract image URL (try multiple common locations)
    image_url = None
    if hasattr(feed_info, 'image') and feed_info.image:
        image_url = feed_info.image.get('href') or feed_info.image.get('url')
    if not image_url and hasattr(feed_info, 'itunes_image'):
        image_url = feed_info.itunes_image.get('href')

    # Extract author
    author = None
    if hasattr(feed_info, 'author'):
        author = feed_info.author
    elif hasattr(feed_info, 'itunes_author'):
        author = feed_info.itunes_author

    return {
        'title': feed_info.get('title', 'Unknown Podcast'),
        'description': feed_info.get('subtitle') or feed_info.get('summary') or feed_info.get('description'),
        'image_url': image_url,
        'author': author,
    }


def extract_audio_url(entry: feedparser.FeedParserDict) -> Optional[str]:
    """
    Extract the audio URL from an RSS entry.

    Args:
        entry: A single RSS feed entry.

    Returns:
        The audio URL or None if not found.
    """
    # Check enclosures first (standard podcast format)
    for enclosure in entry.get("enclosures", []):
        if enclosure.get("type", "").startswith("audio/"):
            return enclosure.get("href")

    # Fallback to links
    for link in entry.get("links", []):
        if link.get("type", "").startswith("audio/"):
            return link.get("href")

    return None


def get_episode_guid(entry: feedparser.FeedParserDict) -> str:
    """
    Get a unique identifier for an episode.

    Args:
        entry: A single RSS feed entry.

    Returns:
        The GUID or a fallback identifier.
    """
    # Prefer the guid field
    if entry.get("id"):
        return entry.get("id")

    # Fallback to link
    if entry.get("link"):
        return entry.get("link")

    # Last resort: title + published date
    return f"{entry.get('title', '')}-{entry.get('published', '')}"


def get_episode_description(entry: feedparser.FeedParserDict) -> Optional[str]:
    """Extract episode description/show notes."""
    # Try content first (usually has full HTML notes)
    if entry.get("content"):
        for content in entry.content:
            if content.get("value"):
                return content.value

    # Fall back to summary
    if entry.get("summary"):
        return entry.summary

    # Try description
    if entry.get("description"):
        return entry.description

    return None


def get_episode_duration(entry: feedparser.FeedParserDict) -> Optional[int]:
    """Extract episode duration in seconds."""
    # Try itunes:duration
    duration_str = entry.get("itunes_duration")
    if duration_str:
        try:
            # Can be seconds as int, or HH:MM:SS / MM:SS format
            if ":" in str(duration_str):
                parts = str(duration_str).split(":")
                if len(parts) == 3:
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                elif len(parts) == 2:
                    return int(parts[0]) * 60 + int(parts[1])
            else:
                return int(duration_str)
        except (ValueError, TypeError):
            pass
    return None


def get_episode_image(entry: feedparser.FeedParserDict) -> Optional[str]:
    """Extract episode-specific artwork URL."""
    # Try itunes:image
    if hasattr(entry, "itunes_image") and entry.itunes_image:
        return entry.itunes_image.get("href")

    # Try image tag
    if entry.get("image"):
        img = entry.image
        if isinstance(img, dict):
            return img.get("href") or img.get("url")

    return None


def ingest_feed(feed_id: int, session: Optional[Session] = None) -> List[Episode]:
    """
    Ingest episodes from a feed, creating new Episode records for new GUIDs.

    Crucial Logic:
    - If Feed.auto_process is True, new episodes get status QUEUED
    - If Feed.auto_process is False, new episodes get status DISCOVERED
    - Only NEW episodes are affected by this flag
    - Existing episodes are never re-queued when toggling the flag

    Args:
        feed_id: The ID of the feed to ingest.
        session: Optional database session (creates one if not provided).

    Returns:
        List of newly created Episode objects.
    """
    close_session = False
    if session is None:
        session = get_session()
        close_session = True

    try:
        # Get the feed
        feed = session.get(Feed, feed_id)
        if not feed:
            logger.error(f"Feed with id {feed_id} not found")
            return []

        # Parse the RSS
        parsed = parse_rss(feed.rss_url)
        if not parsed:
            return []

        # Get existing episode GUIDs for this feed
        existing_guids = set(
            session.exec(
                select(Episode.guid).where(Episode.feed_id == feed_id)
            ).all()
        )

        new_episodes = []

        for entry in parsed.entries:
            guid = get_episode_guid(entry)

            # Skip if we already have this episode
            if guid in existing_guids:
                continue

            audio_url = extract_audio_url(entry)
            if not audio_url:
                logger.warning(f"No audio URL found for entry: {entry.get('title', 'Unknown')}")
                continue

            # Determine status based on auto_process flag
            # This ONLY applies to new episodes being discovered right now
            status = EpisodeStatus.QUEUED if feed.auto_process else EpisodeStatus.DISCOVERED

            # Parse publish date
            published_at = None
            if entry.get("published"):
                try:
                    published_at = parsedate_to_datetime(entry.published)
                except (ValueError, TypeError):
                    pass

            episode = Episode(
                feed_id=feed_id,
                guid=guid,
                status=status,
                title=entry.get("title", "Untitled Episode"),
                audio_url=audio_url,
                description=get_episode_description(entry),
                duration=get_episode_duration(entry),
                image_url=get_episode_image(entry),
                published_at=published_at,
            )

            session.add(episode)
            new_episodes.append(episode)
            logger.info(f"New episode discovered: {episode.title} (status: {status.value})")

        if new_episodes:
            session.commit()
            # Refresh to get IDs
            for ep in new_episodes:
                session.refresh(ep)

        return new_episodes

    finally:
        if close_session:
            session.close()


def ingest_all_feeds() -> dict:
    """
    Ingest episodes from all feeds in the database.

    Returns:
        Dictionary mapping feed_id to list of new episodes.
    """
    with get_session() as session:
        feeds = session.exec(select(Feed)).all()

        results = {}
        for feed in feeds:
            new_episodes = ingest_feed(feed.id, session)
            if new_episodes:
                results[feed.id] = new_episodes

        return results
