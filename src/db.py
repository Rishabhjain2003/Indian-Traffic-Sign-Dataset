"""
Database Layer — PostgreSQL operations via psycopg2.

Manages schema initialisation, video upserts, and batched insertion of
detected frames and their bounding-box detections.

Includes a ``discovered_videos`` table that acts as a persistent,
DB-backed work queue with status tracking (pending → processing →
completed / failed).  Each row is atomically "claimed" by a worker
thread using ``SELECT … FOR UPDATE SKIP LOCKED``.
"""

from __future__ import annotations

import logging
import threading

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

# ── SQL — Schema ─────────────────────────────────────────

_CREATE_VIDEOS = """
CREATE TABLE IF NOT EXISTS videos (
    id              SERIAL PRIMARY KEY,
    url             TEXT NOT NULL UNIQUE,
    youtube_id      TEXT,
    title           TEXT,
    uploader        TEXT,
    upload_date     TEXT,
    duration_sec    INTEGER,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
"""

_CREATE_DETECTED_FRAMES = """
CREATE TABLE IF NOT EXISTS detected_frames (
    id              SERIAL PRIMARY KEY,
    video_id        INTEGER REFERENCES videos(id) ON DELETE CASCADE,
    frame_number    INTEGER NOT NULL,
    timestamp_sec   FLOAT NOT NULL,
    frame_path      TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
"""

_CREATE_DETECTIONS = """
CREATE TABLE IF NOT EXISTS detections (
    id              SERIAL PRIMARY KEY,
    frame_id        INTEGER REFERENCES detected_frames(id) ON DELETE CASCADE,
    class_id        INTEGER,
    class_name      TEXT,
    confidence      FLOAT,
    bbox_x1         FLOAT,
    bbox_y1         FLOAT,
    bbox_x2         FLOAT,
    bbox_y2         FLOAT
);
"""

_CREATE_DISCOVERED_VIDEOS = """
CREATE TABLE IF NOT EXISTS discovered_videos (
    id              SERIAL PRIMARY KEY,
    youtube_id      TEXT NOT NULL UNIQUE,
    url             TEXT NOT NULL,
    title           TEXT,
    channel         TEXT,
    length_sec      INTEGER,
    keyword         TEXT,
    status          TEXT NOT NULL DEFAULT 'pending',
    error_message   TEXT,
    discovered_at   TIMESTAMPTZ DEFAULT NOW(),
    processed_at    TIMESTAMPTZ
);
"""

_CREATE_DISCOVERED_IDX_STATUS = """
CREATE INDEX IF NOT EXISTS idx_discovered_videos_status
ON discovered_videos (status);
"""

_CREATE_DISCOVERED_IDX_YT_ID = """
CREATE INDEX IF NOT EXISTS idx_discovered_videos_youtube_id
ON discovered_videos (youtube_id);
"""

# ── SQL — Discovered-videos queue ────────────────────────

_INSERT_DISCOVERED = """
INSERT INTO discovered_videos (youtube_id, url, title, channel, length_sec, keyword)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (youtube_id) DO NOTHING;
"""

_CLAIM_NEXT = """
UPDATE discovered_videos
SET    status = 'processing'
WHERE  id = (
    SELECT id FROM discovered_videos
    WHERE  status = 'pending'
    ORDER  BY id
    LIMIT  1
    FOR UPDATE SKIP LOCKED
)
RETURNING id, youtube_id, url, title, channel, length_sec;
"""

_MARK_STATUS = """
UPDATE discovered_videos
SET    status = %s,
       error_message = %s,
       processed_at  = CASE WHEN %s IN ('completed', 'failed') THEN NOW() ELSE processed_at END
WHERE  youtube_id = %s;
"""

_COUNT_BY_STATUS = """
SELECT status, COUNT(*) FROM discovered_videos GROUP BY status ORDER BY status;
"""

_RESET_STALE_PROCESSING = """
UPDATE discovered_videos
SET    status = 'pending'
WHERE  status = 'processing';
"""

# ── SQL — Videos / Frames / Detections ───────────────────

_UPSERT_VIDEO = """
INSERT INTO videos (url, youtube_id, title, uploader, upload_date, duration_sec)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (url) DO UPDATE SET
    youtube_id   = EXCLUDED.youtube_id,
    title        = EXCLUDED.title,
    uploader     = EXCLUDED.uploader,
    upload_date  = EXCLUDED.upload_date,
    duration_sec = EXCLUDED.duration_sec
RETURNING id;
"""

_INSERT_FRAME = """
INSERT INTO detected_frames (video_id, frame_number, timestamp_sec, frame_path)
VALUES (%s, %s, %s, %s)
RETURNING id;
"""

_INSERT_DETECTION = """
INSERT INTO detections (frame_id, class_id, class_name, confidence,
                        bbox_x1, bbox_y1, bbox_x2, bbox_y2)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
"""


class Database:
    """Thread-safe PostgreSQL wrapper for the traffic sign pipeline.

    Each ``Database`` instance owns its own connection.  In a
    multi-threaded setup, create one instance per thread.
    """

    def __init__(self, database_url: str) -> None:
        self._url = database_url
        self._lock = threading.Lock()
        logger.info("Connecting to PostgreSQL …")
        self.conn = psycopg2.connect(database_url)
        self.conn.autocommit = False
        self._init_schema()
        logger.info("Database ready — schema initialised")

    # ── Schema ───────────────────────────────────────────

    def _init_schema(self) -> None:
        """Create tables and indexes if they don't already exist."""
        with self.conn.cursor() as cur:
            cur.execute(_CREATE_VIDEOS)
            cur.execute(_CREATE_DETECTED_FRAMES)
            cur.execute(_CREATE_DETECTIONS)
            cur.execute(_CREATE_DISCOVERED_VIDEOS)
            cur.execute(_CREATE_DISCOVERED_IDX_STATUS)
            cur.execute(_CREATE_DISCOVERED_IDX_YT_ID)
        self.conn.commit()

    # ── Discovered-videos queue ──────────────────────────

    def save_discovered(self, video: dict, keyword: str | None = None) -> bool:
        """Insert a newly discovered video into the queue.

        Returns True if the video was inserted (new), False if it already
        existed (duplicate youtube_id).
        """
        with self._lock:
            with self.conn.cursor() as cur:
                cur.execute(
                    _INSERT_DISCOVERED,
                    (
                        video["youtube_id"],
                        video["url"],
                        video.get("title"),
                        video.get("channel"),
                        video.get("length_sec"),
                        keyword,
                    ),
                )
                inserted = cur.rowcount > 0
            self.conn.commit()
        return inserted

    def claim_next_video(self) -> dict | None:
        """Atomically claim the next pending video for processing.

        Uses ``FOR UPDATE SKIP LOCKED`` so multiple worker threads can
        call this concurrently without contention.

        Returns a dict with video info, or None if no pending videos.
        """
        with self._lock:
            with self.conn.cursor() as cur:
                cur.execute(_CLAIM_NEXT)
                row = cur.fetchone()
            self.conn.commit()

        if row is None:
            return None

        return {
            "db_id": row[0],
            "youtube_id": row[1],
            "url": row[2],
            "title": row[3],
            "channel": row[4],
            "length_sec": row[5],
        }

    def mark_discovered_status(
        self,
        youtube_id: str,
        status: str,
        error_message: str | None = None,
    ) -> None:
        """Update the status of a discovered video."""
        with self._lock:
            with self.conn.cursor() as cur:
                cur.execute(_MARK_STATUS, (status, error_message, status, youtube_id))
            self.conn.commit()

    def get_queue_stats(self) -> dict[str, int]:
        """Return a dict of {status: count} for the discovered_videos table."""
        with self.conn.cursor() as cur:
            cur.execute(_COUNT_BY_STATUS)
            return dict(cur.fetchall())

    def reset_stale_processing(self) -> int:
        """Reset any 'processing' rows back to 'pending'.

        Call this at startup to recover from a previous crash that left
        rows stuck in 'processing' state.
        """
        with self.conn.cursor() as cur:
            cur.execute(_RESET_STALE_PROCESSING)
            count = cur.rowcount
        self.conn.commit()
        if count:
            logger.info("Reset %d stale 'processing' rows to 'pending'", count)
        return count

    # ── Video ────────────────────────────────────────────

    def upsert_video(self, meta: dict) -> int:
        """Insert or update a video row and return its ``videos.id``.

        Parameters
        ----------
        meta:
            Dict with keys ``url``, ``youtube_id``, ``title``, ``uploader``,
            ``upload_date``, ``duration_sec``.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                _UPSERT_VIDEO,
                (
                    meta["url"],
                    meta.get("youtube_id"),
                    meta.get("title"),
                    meta.get("uploader"),
                    meta.get("upload_date"),
                    meta.get("duration_sec"),
                ),
            )
            video_id: int = cur.fetchone()[0]
        self.conn.commit()
        logger.info("Upserted video id=%d  url=%s", video_id, meta["url"])
        return video_id

    # ── Detections ───────────────────────────────────────

    def save_results(
        self,
        video_db_id: int,
        detected_frames: list[dict],
    ) -> None:
        """Persist all detected frames and their detections in one transaction.

        Parameters
        ----------
        video_db_id:
            The ``videos.id`` foreign key.
        detected_frames:
            List of frame-info dicts, each containing a ``detections`` key
            with a list of detection dicts.
        """
        if not detected_frames:
            logger.info("No detections to save for video id=%d", video_db_id)
            return

        inserted_frames = 0
        inserted_dets = 0

        try:
            with self.conn.cursor() as cur:
                for frame in detected_frames:
                    # Insert the detected_frame row
                    cur.execute(
                        _INSERT_FRAME,
                        (
                            video_db_id,
                            frame["frame_number"],
                            frame["timestamp_sec"],
                            str(frame.get("frame_path", "")),
                        ),
                    )
                    frame_db_id: int = cur.fetchone()[0]
                    inserted_frames += 1

                    # Insert each detection for this frame
                    for det in frame["detections"]:
                        bbox = det["bbox"]  # [x1, y1, x2, y2]
                        cur.execute(
                            _INSERT_DETECTION,
                            (
                                frame_db_id,
                                det["class_id"],
                                det["class_name"],
                                det["confidence"],
                                bbox[0],
                                bbox[1],
                                bbox[2],
                                bbox[3],
                            ),
                        )
                        inserted_dets += 1

            self.conn.commit()
            logger.info(
                "Saved %d frames, %d detections for video id=%d",
                inserted_frames,
                inserted_dets,
                video_db_id,
            )

        except Exception:
            self.conn.rollback()
            logger.exception(
                "Failed to save results for video id=%d — rolled back",
                video_db_id,
            )
            raise

    # ── Lifecycle ────────────────────────────────────────

    def close(self) -> None:
        """Commit any pending work and close the connection."""
        if self.conn and not self.conn.closed:
            try:
                self.conn.commit()
            except Exception:
                pass
            self.conn.close()
            logger.info("Database connection closed")
