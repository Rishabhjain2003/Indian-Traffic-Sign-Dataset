#!/usr/bin/env python3
"""
Pipeline Orchestrator — Decoupled producer-consumer architecture.

Architecture::

    ┌──────────────┐     ┌───────────────────┐     ┌─────────────────┐
    │   Scraper    │────▸│  discovered_videos │────▸│  Worker Thread  │
    │  (Producer)  │     │   (DB + Queue)     │     │  (Consumer ×N)  │
    └──────────────┘     └───────────────────┘     └─────────────────┘
                              ▲                          │
                              │     download → extract → detect → store
                              └──────── mark completed ──┘

The **producer** thread discovers videos via YouTube search and
immediately saves each one to the ``discovered_videos`` DB table
(status = 'pending').  It also pushes a notification onto an
in-memory ``queue.Queue``.

**N worker threads** (consumers) pull from the queue, claim the next
pending video atomically (``FOR UPDATE SKIP LOCKED``), and process
it through the full pipeline: download → extract → detect → store →
cleanup.

On startup, any rows stuck in 'processing' from a previous crash are
reset to 'pending', and all existing 'pending' rows are loaded into
the queue for processing.

Usage::

    # Search mode — producer + 2 worker threads (default)
    python pipeline.py --search

    # More workers for faster processing
    python pipeline.py --search --workers 4

    # Only run workers (no new search, process existing pending queue)
    python pipeline.py --workers-only

    # Process a single URL (bypasses queue, runs inline)
    python pipeline.py --url "https://www.youtube.com/watch?v=..."

    # Process URLs from file (adds to queue + starts workers)
    python pipeline.py --urls urls.txt
"""

from __future__ import annotations

import argparse
import logging
import queue
import re
import shutil
import signal
import sys
import threading
import time
from pathlib import Path

from config.settings import settings
from src.db import Database
from src.detector import TrafficSignDetector
from src.downloader import download_video, load_urls
from src.extractor import extract_frames
from src.scraper import load_keywords, search_videos

# ── Logging ──────────────────────────────────────────────

LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  [%(threadName)s]  %(message)s"
LOG_DATE = "%Y-%m-%d %H:%M:%S"


def _setup_logging() -> None:
    """Configure dual logging to stdout and ``pipeline.log``."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE))
    root.addHandler(console)

    fh = logging.FileHandler("pipeline.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE))
    root.addHandler(fh)


logger = logging.getLogger("pipeline")


# ── CLI ──────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Indian Traffic Sign Detection Pipeline (Producer-Consumer)",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--search",
        action="store_true",
        help="Search YouTube using keywords + start worker threads",
    )
    source.add_argument(
        "--workers-only",
        action="store_true",
        help="Skip search — only process existing pending videos in the DB queue",
    )
    source.add_argument(
        "--url",
        type=str,
        help="Process a single YouTube URL (inline, no threads)",
    )
    source.add_argument(
        "--urls",
        type=str,
        help="Add URLs from a file to the queue and start workers",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        default=None,
        help="Path to keywords file (default: combinatorial_keywords.txt)",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=None,
        help="Max results per keyword in search mode",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads (default: from settings)",
    )
    return parser.parse_args()


# ── Helpers ──────────────────────────────────────────────


def _extract_video_id(url: str) -> str | None:
    """Best-effort extraction of the YouTube video ID from a URL."""
    patterns = [
        r"(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None


def _cleanup_video(video_path: Path | None, video_id: str) -> None:
    """Delete the video file and extracted frames from disk."""
    if video_path and video_path.exists():
        try:
            video_path.unlink()
            logger.info("Deleted video file: %s", video_path)
        except Exception:
            logger.warning("Could not delete video: %s", video_path, exc_info=True)

    frames_dir = settings.frames_dir / video_id
    if frames_dir.exists():
        try:
            shutil.rmtree(frames_dir)
            logger.info("Deleted frames directory: %s", frames_dir)
        except Exception:
            logger.warning("Could not delete frames: %s", frames_dir, exc_info=True)


# ── Producer ─────────────────────────────────────────────


def producer_thread(
    db: Database,
    work_queue: queue.Queue,
    keywords: list[str],
    max_results: int,
    stop_event: threading.Event,
) -> None:
    """Discover videos via YouTube search and enqueue them for processing.

    Each discovered video is immediately saved to the ``discovered_videos``
    table.  A lightweight signal (youtube_id) is put on ``work_queue`` so
    that worker threads wake up and claim it.
    """
    new_count = 0

    def on_discovered(video: dict, keyword: str) -> None:
        nonlocal new_count
        inserted = db.save_discovered(video, keyword=keyword)
        if inserted:
            new_count += 1
            work_queue.put(video["youtube_id"])

    try:
        search_videos(
            keywords,
            max_results_per_keyword=max_results,
            delay_between_queries=settings.search_delay,
            on_discovered=on_discovered,
            stop_event=stop_event,
        )
    except Exception:
        logger.exception("Producer thread crashed")
    finally:
        logger.info(
            "Producer finished — %d new videos added to queue", new_count
        )


# ── Worker (Consumer) ────────────────────────────────────


def worker_thread(
    worker_id: int,
    db: Database,
    detector: TrafficSignDetector | None,
    work_queue: queue.Queue,
    stop_event: threading.Event,
    stats: dict,
    stats_lock: threading.Lock,
) -> None:
    """Consume videos from the queue and process them.

    Each iteration:
    1. Wait for a signal on the queue (or poll every 5s)
    2. Atomically claim the next pending video from the DB
    3. Download → extract frames → detect → store → cleanup
    4. Mark the video as completed or failed
    """
    thread_name = f"Worker-{worker_id}"
    logger.info("[%s] Started", thread_name)

    while not stop_event.is_set():
        # Wait for work (with timeout so we can check stop_event)
        try:
            work_queue.get(timeout=5.0)
        except queue.Empty:
            # No signal — but there might be pending rows from a previous
            # run or another producer.  Try to claim one anyway.
            pass

        # Atomically claim the next pending video
        video = db.claim_next_video()
        if video is None:
            continue  # Nothing to process right now

        youtube_id = video["youtube_id"]
        url = video["url"]

        try:
            logger.info(
                "[%s] Processing %s — %s",
                thread_name,
                youtube_id,
                video.get("title") or "untitled",
            )

            # ── 1. Download ──────────────────────────────
            meta = download_video(url, settings)
            if meta is None:
                db.mark_discovered_status(youtube_id, "failed", "download_failed")
                with stats_lock:
                    stats["failed"] += 1
                continue

            video_path = meta.get("filepath")
            duration = meta.get("duration_sec", 0) or 0

            # ── 2. Check duration cap ─────────────────────
            if duration > settings.max_video_duration_sec:
                logger.info(
                    "[%s] Skipping %s — too long (%ds > %ds)",
                    thread_name,
                    youtube_id,
                    duration,
                    settings.max_video_duration_sec,
                )
                db.mark_discovered_status(youtube_id, "completed", "too_long")
                if video_path and settings.delete_video_after_processing:
                    _cleanup_video(video_path, youtube_id)
                with stats_lock:
                    stats["completed"] += 1
                continue

            # ── 3. Upsert into videos table ──────────────
            video_db_id = db.upsert_video(meta)

            # ── 4. Extract frames ────────────────────────
            if video_path is None or not Path(video_path).exists():
                db.mark_discovered_status(youtube_id, "failed", "video_file_missing")
                with stats_lock:
                    stats["failed"] += 1
                continue

            frame_infos = extract_frames(Path(video_path), youtube_id, settings)

            if not frame_infos:
                logger.warning("No frames extracted for %s", youtube_id)
                db.mark_discovered_status(youtube_id, "completed", "no_frames")
                if settings.delete_video_after_processing:
                    _cleanup_video(video_path, youtube_id)
                with stats_lock:
                    stats["completed"] += 1
                continue

            # ── 5. Detect traffic signs ──────────────────
            if detector is None:
                logger.warning("No YOLO detector — skipping detection for %s", youtube_id)
                db.mark_discovered_status(youtube_id, "completed", "no_detector")
                if settings.delete_video_after_processing:
                    _cleanup_video(video_path, youtube_id)
                with stats_lock:
                    stats["completed"] += 1
                continue

            detected_frames = detector.detect_frames(frame_infos, youtube_id)

            # ── 6. Store results ─────────────────────────
            db.save_results(video_db_id, detected_frames)

            # ── 7. Mark completed ────────────────────────
            db.mark_discovered_status(youtube_id, "completed")

            # ── 8. Cleanup ───────────────────────────────
            if settings.delete_video_after_processing:
                _cleanup_video(video_path, youtube_id)

            with stats_lock:
                stats["completed"] += 1

            logger.info(
                "[%s] ✓ Finished %s — %d frames with detections",
                thread_name,
                youtube_id,
                len(detected_frames),
            )

        except Exception as exc:
            logger.exception("[%s] Failed processing %s", thread_name, youtube_id)
            try:
                db.mark_discovered_status(
                    youtube_id, "failed", str(exc)[:500]
                )
            except Exception:
                pass
            with stats_lock:
                stats["failed"] += 1

    logger.info("[%s] Stopped", thread_name)


# ── Inline processing (single URL mode) ─────────────────


def process_single_url(url: str, db: Database, detector: TrafficSignDetector | None) -> None:
    """Process a single URL inline (no threading)."""
    video_id = _extract_video_id(url)

    logger.info("=" * 60)
    logger.info("Processing: %s", url)
    logger.info("=" * 60)

    meta = download_video(url, settings)
    if meta is None:
        logger.error("Download failed for %s", url)
        return

    video_id = meta["youtube_id"]
    video_path = meta.get("filepath")
    video_db_id = db.upsert_video(meta)

    if video_path is None or not Path(video_path).exists():
        logger.error("Video file not found for %s", video_id)
        return

    frame_infos = extract_frames(Path(video_path), video_id, settings)
    if not frame_infos:
        logger.warning("No frames for %s", video_id)
        if settings.delete_video_after_processing:
            _cleanup_video(video_path, video_id)
        return

    if detector:
        detected_frames = detector.detect_frames(frame_infos, video_id)
        db.save_results(video_db_id, detected_frames)
        logger.info("✓ %s — %d frames with detections", video_id, len(detected_frames))
    else:
        logger.warning("No YOLO detector — skipping detection")

    if settings.delete_video_after_processing:
        _cleanup_video(video_path, video_id)


# ── Main ─────────────────────────────────────────────────


def main() -> None:
    _setup_logging()
    args = _parse_args()

    num_workers = args.workers or settings.num_workers

    # ── Single URL mode (no threading) ───────────────────
    if args.url:
        db = Database(settings.database_url)
        try:
            detector = TrafficSignDetector(settings)
        except FileNotFoundError as exc:
            logger.warning("YOLO unavailable: %s", exc)
            detector = None
        try:
            process_single_url(args.url.strip(), db, detector)
        finally:
            db.close()
        return

    # ── Threaded modes ───────────────────────────────────
    stop_event = threading.Event()
    work_queue: queue.Queue = queue.Queue()

    # Graceful shutdown on Ctrl+C
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received — stopping threads …")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # ── DB setup (main thread connection) ────────────────
    main_db = Database(settings.database_url)

    # Reset any rows stuck in 'processing' from a previous crash
    main_db.reset_stale_processing()

    # Load existing pending videos into the queue
    stats_before = main_db.get_queue_stats()
    pending_count = stats_before.get("pending", 0)
    if pending_count:
        logger.info(
            "Found %d pending videos from previous runs — adding to queue",
            pending_count,
        )
        for _ in range(pending_count):
            work_queue.put("resume")

    # ── Add URLs from file (--urls mode) ─────────────────
    if args.urls:
        urls = load_urls(args.urls)
        added = 0
        for u in urls:
            vid_id = _extract_video_id(u)
            if vid_id:
                inserted = main_db.save_discovered(
                    {"url": u, "youtube_id": vid_id}, keyword="manual"
                )
                if inserted:
                    work_queue.put(vid_id)
                    added += 1
        logger.info("Added %d new URLs from %s to queue", added, args.urls)

    # ── Load YOLO detector ───────────────────────────────
    detector: TrafficSignDetector | None = None
    try:
        detector = TrafficSignDetector(settings)
    except FileNotFoundError as exc:
        logger.warning("YOLO unavailable: %s — detection will be skipped", exc)

    # ── Start worker threads ─────────────────────────────
    stats = {"completed": 0, "failed": 0}
    stats_lock = threading.Lock()

    workers: list[threading.Thread] = []
    for i in range(num_workers):
        # Each worker gets its own DB connection
        worker_db = Database(settings.database_url)
        t = threading.Thread(
            target=worker_thread,
            args=(i, worker_db, detector, work_queue, stop_event, stats, stats_lock),
            name=f"Worker-{i}",
            daemon=True,
        )
        t.start()
        workers.append(t)

    logger.info("Started %d worker threads", num_workers)

    # ── Start producer thread (search mode) ──────────────
    producer: threading.Thread | None = None
    if args.search:
        kw_file = args.keywords or str(settings.keywords_file)
        max_res = args.max_results or settings.max_results_per_keyword

        keywords = load_keywords(kw_file)
        if not keywords:
            logger.error("No keywords — exiting")
            stop_event.set()
        else:
            producer = threading.Thread(
                target=producer_thread,
                args=(main_db, work_queue, keywords, max_res, stop_event),
                name="Producer",
                daemon=True,
            )
            producer.start()
            logger.info("Started producer thread — searching %d keywords", len(keywords))

    # ── Wait for completion ──────────────────────────────
    try:
        if producer:
            producer.join()
            logger.info("Producer finished — waiting for workers to drain queue …")

        # Wait for the queue to drain
        while not stop_event.is_set():
            queue_stats = main_db.get_queue_stats()
            pending = queue_stats.get("pending", 0)
            processing = queue_stats.get("processing", 0)

            if pending == 0 and processing == 0:
                logger.info("All videos processed — shutting down")
                stop_event.set()
                break

            logger.info(
                "Queue status — pending: %d, processing: %d, completed: %d, failed: %d",
                pending,
                processing,
                queue_stats.get("completed", 0),
                queue_stats.get("failed", 0),
            )
            time.sleep(10)

    except KeyboardInterrupt:
        logger.info("Ctrl+C — shutting down …")
        stop_event.set()

    # ── Wait for workers to finish ───────────────────────
    for t in workers:
        t.join(timeout=30)

    # ── Final stats ──────────────────────────────────────
    final_stats = main_db.get_queue_stats()
    logger.info("=" * 60)
    logger.info("Pipeline complete")
    logger.info(
        "  Completed: %d  |  Failed: %d  |  Pending: %d",
        final_stats.get("completed", 0),
        final_stats.get("failed", 0),
        final_stats.get("pending", 0),
    )
    logger.info("=" * 60)

    main_db.close()


if __name__ == "__main__":
    main()
