"""
Frame Extractor — pull frames from a video file using ffmpeg.

Uses a subprocess call to ``ffmpeg`` for maximum reliability.
Supports configurable FPS and frame stride (keep every Nth frame).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from config.settings import Settings

logger = logging.getLogger(__name__)


def extract_frames(
    video_path: Path,
    video_id: str,
    settings: Settings,
) -> list[dict]:
    """Extract frames from *video_path* at the configured FPS.

    Parameters
    ----------
    video_path:
        Path to the source video file.
    video_id:
        YouTube video ID, used to organise output into a subdirectory.
    settings:
        Pipeline configuration instance.

    Returns
    -------
    list[dict]
        Each dict contains ``frame_number`` (int), ``frame_path`` (Path),
        and ``timestamp_sec`` (float).
    """
    out_dir = settings.frames_dir / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Resumability: skip if frames already exist ───────
    existing_frames = sorted(out_dir.glob("frame_*.jpg"))
    if existing_frames:
        logger.info(
            "Frames already exist for %s (%d files) — skipping extraction",
            video_id,
            len(existing_frames),
        )
        return _build_frame_list(existing_frames, settings.fps)

    # ── Verify ffmpeg is available ───────────────────────
    if not shutil.which("ffmpeg"):
        logger.error("ffmpeg not found on PATH — cannot extract frames")
        return []

    # ── Run ffmpeg ───────────────────────────────────────
    output_pattern = str(out_dir / "frame_%06d.jpg")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",
        "-i", str(video_path),
        "-vf", f"fps={settings.fps}",
        "-q:v", "2",
        output_pattern,
    ]

    logger.info("Extracting frames from %s at %d FPS …", video_id, settings.fps)
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        logger.error(
            "ffmpeg failed for %s — stderr:\n%s", video_id, exc.stderr
        )
        return []

    # ── Apply frame stride ───────────────────────────────
    all_frames = sorted(out_dir.glob("frame_*.jpg"))
    if settings.frame_stride > 1:
        kept, removed = _apply_stride(all_frames, settings.frame_stride)
        logger.info(
            "Frame stride %d: kept %d / %d frames, removed %d",
            settings.frame_stride,
            len(kept),
            len(all_frames),
            removed,
        )
        all_frames = kept

    frame_list = _build_frame_list(all_frames, settings.fps)
    logger.info(
        "Extracted %d frames for %s", len(frame_list), video_id
    )
    return frame_list


# ── Helpers ──────────────────────────────────────────────


def _frame_number_from_path(path: Path) -> int:
    """Parse the frame number from a filename like ``frame_000123.jpg``.

    ffmpeg's ``%06d`` pattern starts numbering at 1, so we subtract 1 to
    make frame numbers zero-based (consistent with timestamp calculation).
    """
    stem = path.stem  # e.g. "frame_000123"
    num_str = stem.split("_", maxsplit=1)[1]
    return int(num_str) - 1  # zero-based


def _build_frame_list(frames: list[Path], fps: int) -> list[dict]:
    """Build the standard frame-info dicts from a sorted list of paths."""
    result: list[dict] = []
    for fp in frames:
        fnum = _frame_number_from_path(fp)
        result.append(
            {
                "frame_number": fnum,
                "frame_path": fp,
                "timestamp_sec": round(fnum / fps, 4),
            }
        )
    return result


def _apply_stride(frames: list[Path], stride: int) -> tuple[list[Path], int]:
    """Keep every *stride*-th frame, delete the rest from disk.

    Returns the list of kept frame paths and the count of removed files.
    """
    kept: list[Path] = []
    removed = 0
    for idx, fp in enumerate(frames):
        if idx % stride == 0:
            kept.append(fp)
        else:
            fp.unlink(missing_ok=True)
            removed += 1
    return kept, removed
