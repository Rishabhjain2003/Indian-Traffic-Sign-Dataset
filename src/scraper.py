"""
YouTube Search Scraper — discover dashcam video URLs using the InnerTube API.

Uses ``pytubefix.Search`` to query YouTube for each keyword loaded from a
text file (e.g. ``combinatorial_keywords.txt``).  Each discovered video
is immediately persisted to the database via a callback, decoupling
discovery from processing.
"""

from __future__ import annotations

import logging
import signal
import time
from pathlib import Path
from typing import Callable

from pytubefix import Search

logger = logging.getLogger(__name__)


def load_keywords(filepath: str | Path) -> list[str]:
    """Read a keyword file and return a list of non-blank, non-comment lines.

    Parameters
    ----------
    filepath:
        Path to a ``.txt`` file with one search query per line.
        Lines starting with ``#`` and blank lines are ignored.
    """
    path = Path(filepath)
    if not path.exists():
        logger.error("Keyword file not found: %s", path)
        return []

    keywords: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            keywords.append(stripped)
    logger.info("Loaded %d keywords from %s", len(keywords), path)
    return keywords


def search_videos(
    keywords: list[str],
    max_results_per_keyword: int = 5,
    delay_between_queries: float = 1.0,
    on_discovered: Callable[[dict, str], None] | None = None,
    stop_event=None,
    search_timeout: float = 30.0,
) -> int:
    """Search YouTube for each keyword and push results via callback.

    Parameters
    ----------
    keywords:
        List of search query strings.
    max_results_per_keyword:
        Maximum number of video results to collect per keyword.
    delay_between_queries:
        Seconds to wait between successive search requests to avoid
        rate-limiting.
    on_discovered:
        Callback ``(video_dict, keyword) -> None`` called for each
        unique video found.  If None, videos are only counted.
    stop_event:
        A ``threading.Event`` that, when set, signals the search to
        stop early.
    search_timeout:
        Maximum seconds to spend on a single keyword search before
        moving on.

    Returns
    -------
    int
        Total number of unique videos discovered.
    """
    seen_ids: set[str] = set()
    total_discovered = 0

    total = len(keywords)
    logger.info(
        "Starting YouTube search — %d keywords, max %d results each",
        total,
        max_results_per_keyword,
    )

    for idx, keyword in enumerate(keywords, start=1):
        # Check for early stop
        if stop_event and stop_event.is_set():
            logger.info("Stop signal received — halting search at keyword %d/%d", idx, total)
            break

        try:
            logger.info("  [%d/%d] Searching: %s", idx, total, keyword)

            # Use signal-based timeout for the search (main thread only)
            # For worker threads, we rely on pytubefix's internal timeouts
            s = Search(keyword)

            collected = 0
            for video in s.videos:
                if collected >= max_results_per_keyword:
                    break

                vid_id = video.video_id
                if vid_id in seen_ids:
                    continue

                # Build video dict with safe attribute access
                video_info = _safe_video_info(video)
                if video_info is None:
                    continue

                seen_ids.add(vid_id)
                total_discovered += 1
                collected += 1

                # Fire callback — this saves to DB immediately
                if on_discovered:
                    try:
                        on_discovered(video_info, keyword)
                    except Exception:
                        logger.exception(
                            "Callback failed for %s — continuing", vid_id
                        )

            logger.info(
                "    Found %d new videos (total unique: %d)",
                collected,
                total_discovered,
            )

        except Exception:
            logger.exception("Search failed for keyword: %s — skipping", keyword)

        # Rate-limit between queries
        if idx < total and delay_between_queries > 0:
            time.sleep(delay_between_queries)

    logger.info(
        "Search complete — %d unique videos discovered from %d keywords",
        total_discovered,
        total,
    )
    return total_discovered


def _safe_video_info(video) -> dict | None:
    """Extract video metadata with graceful error handling.

    Returns None if critical attributes (video_id, watch_url) can't
    be accessed — typically due to BotDetection or restricted content.
    """
    try:
        vid_id = video.video_id
        url = video.watch_url
    except Exception:
        return None

    # Title, author, length may trigger BotDetection — handle gracefully
    title = None
    channel = None
    length_sec = None

    try:
        title = video.title
    except Exception:
        pass

    try:
        channel = video.author
    except Exception:
        pass

    try:
        length_sec = video.length
    except Exception:
        pass

    return {
        "url": url,
        "youtube_id": vid_id,
        "title": title,
        "channel": channel,
        "length_sec": length_sec,
    }
