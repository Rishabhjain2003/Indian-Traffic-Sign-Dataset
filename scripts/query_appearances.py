#!/usr/bin/env python3
"""
Appearance Duration Query — for each detected traffic sign class, find
the video URL and time ranges during which it appeared continuously.

"Continuous appearance" means consecutive frames with the same class where
the gap between timestamps is ≤ 1 second.

Outputs a pretty-printed table to stdout and exports to
``output/appearances.csv``.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import psycopg2
import psycopg2.extras

# ── Configuration ────────────────────────────────────────

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://traffic:traffic@localhost:5432/indian_traffic_signs",
)

OUTPUT_CSV = Path("output") / "appearances.csv"

# Maximum gap (in seconds) between consecutive detections that is still
# treated as the same continuous appearance.
MAX_GAP_SEC = 1.0

# ── SQL ──────────────────────────────────────────────────

_QUERY = """
SELECT
    d.class_name,
    v.url,
    v.youtube_id,
    df.timestamp_sec
FROM detections d
JOIN detected_frames df ON d.frame_id = df.id
JOIN videos v           ON df.video_id = v.id
ORDER BY d.class_name, v.id, df.timestamp_sec;
"""


# ── Core Logic ───────────────────────────────────────────


def fetch_appearances(database_url: str) -> list[dict]:
    """Query the database and group detections into appearance windows.

    Returns a list of dicts with keys: ``class_name``, ``url``,
    ``youtube_id``, ``start_sec``, ``end_sec``, ``duration_sec``.
    """
    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(_QUERY)
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return []

    appearances: list[dict] = []
    current_class: str | None = None
    current_url: str | None = None
    window_start: float = 0.0
    window_end: float = 0.0
    current_youtube_id: str | None = None

    def _flush() -> None:
        """Commit the current window to the results list."""
        if current_class is not None and current_url is not None:
            appearances.append(
                {
                    "class_name": current_class,
                    "url": current_url,
                    "youtube_id": current_youtube_id,
                    "start_sec": round(window_start, 2),
                    "end_sec": round(window_end, 2),
                    "duration_sec": round(window_end - window_start, 2),
                }
            )

    for row in rows:
        cls = row["class_name"]
        url = row["url"]
        yt_id = row["youtube_id"]
        ts = float(row["timestamp_sec"])

        if cls != current_class or url != current_url:
            # New class or new video — flush previous window, start fresh
            _flush()
            current_class = cls
            current_url = url
            current_youtube_id = yt_id
            window_start = ts
            window_end = ts
        elif ts - window_end > MAX_GAP_SEC:
            # Gap too large — flush and start a new window
            _flush()
            window_start = ts
            window_end = ts
        else:
            # Extend the current window
            window_end = ts

    # Don't forget the last window
    _flush()

    return appearances


# ── Output ───────────────────────────────────────────────


def print_table(appearances: list[dict]) -> None:
    """Pretty-print the appearances as an aligned table."""
    if not appearances:
        print("No appearances found.")
        return

    header = (
        f"{'Class Name':<30}  {'Video URL':<55}  "
        f"{'Start (s)':>10}  {'End (s)':>10}  {'Duration (s)':>12}"
    )
    separator = "-" * len(header)

    print(separator)
    print(header)
    print(separator)

    for a in appearances:
        print(
            f"{a['class_name']:<30}  {a['url']:<55}  "
            f"{a['start_sec']:>10.2f}  {a['end_sec']:>10.2f}  "
            f"{a['duration_sec']:>12.2f}"
        )

    print(separator)
    print(f"Total appearance windows: {len(appearances)}")


def export_csv(appearances: list[dict], path: Path) -> None:
    """Write appearances to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "class_name",
        "url",
        "youtube_id",
        "start_sec",
        "end_sec",
        "duration_sec",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(appearances)

    print(f"\nCSV exported to: {path}")


# ── Entry Point ──────────────────────────────────────────


def main() -> None:
    print("Querying traffic sign appearances …\n")
    appearances = fetch_appearances(DATABASE_URL)
    print_table(appearances)
    export_csv(appearances, OUTPUT_CSV)


if __name__ == "__main__":
    main()
