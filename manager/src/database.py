import os
from sqlmodel import SQLModel, create_engine, Session, select
from sqlalchemy import text, inspect

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://podcast:podcast@localhost:5432/podcastpurifier")

engine = create_engine(DATABASE_URL, echo=False)


def migrate_add_slugs() -> None:
    """Add slug column and populate slugs for existing feeds."""
    from .models import Feed, generate_slug

    inspector = inspect(engine)
    columns = [c['name'] for c in inspector.get_columns('feeds')]

    with Session(engine) as session:
        # Check if slug column exists
        if 'slug' not in columns:
            # Add slug column as nullable first
            session.exec(text("ALTER TABLE feeds ADD COLUMN slug VARCHAR"))
            session.commit()

        # Populate slugs for feeds that don't have one
        feeds_without_slug = session.exec(
            select(Feed).where(Feed.slug == None)  # noqa: E711
        ).all()

        for feed in feeds_without_slug:
            base_slug = generate_slug(feed.title)
            slug = base_slug
            counter = 2
            # Ensure uniqueness
            while session.exec(select(Feed).where(Feed.slug == slug)).first():
                slug = f"{base_slug}-{counter}"
                counter += 1
            feed.slug = slug
            session.add(feed)

        session.commit()

        # Make slug column not null and add unique constraint if needed
        if 'slug' not in columns:
            session.exec(text("ALTER TABLE feeds ALTER COLUMN slug SET NOT NULL"))
            session.exec(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_feeds_slug ON feeds (slug)"))
            session.commit()


def init_db() -> None:
    """Initialize database tables."""
    SQLModel.metadata.create_all(engine)
    # Run migrations for existing databases
    try:
        migrate_add_slugs()
    except Exception:
        pass  # Table might not exist yet on fresh install


def get_session() -> Session:
    """Get a database session."""
    return Session(engine)
