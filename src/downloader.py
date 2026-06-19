"""
Video Downloader — download YouTube videos via pytubefix.

Uses pytubefix's stream API to download the best available MP4 stream
(progressive ≤720p, or adaptive ≤1080p with ffmpeg mux).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from pytubefix import YouTube

from config.settings import Settings

logger = logging.getLogger(__name__)


def load_urls(filepath: str | Path) -> list[str]:
    """Read a text file and return a list of non-empty, non-comment URLs.

    Parameters
    ----------
    filepath:
        Path to a ``.txt`` file with one URL per line.
        Lines starting with ``#`` and blank lines are ignored.
    """
    path = Path(filepath)
    if not path.exists():
        logger.error("URL file not found: %s", path)
        return []

    urls: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            urls.append(stripped)
    logger.info("Loaded %d URLs from %s", len(urls), path)
    return urls


def download_video(url: str, settings: Settings) -> dict | None:
    """Download a single YouTube video and return its metadata.

    Attempts to get the best MP4 stream up to the configured max
    resolution.  For adaptive streams (1080p), downloads video + audio
    separately and muxes with ffmpeg.

    Parameters
    ----------
    url:
        Full YouTube URL.
    settings:
        Pipeline configuration instance.

    Returns
    -------
    dict | None
        A dict with keys ``url``, ``youtube_id``, ``title``, ``uploader``,
        ``upload_date``, ``duration_sec``, ``filepath`` on success, or
        ``None`` on failure.
    """
    try:
        yt = YouTube(url)
        video_id = yt.video_id
        out_dir = settings.videos_dir

        # Try to get the best progressive stream first (has audio, max 720p)
        stream = (
            yt.streams
            .filter(progressive=True, file_extension="mp4")
            .order_by("resolution")
            .desc()
            .first()
        )

        filepath: Path | None = None

        if stream:
            # Progressive download — single file with audio
            logger.info(
                "Downloading (progressive %s): %s", stream.resolution, yt.title
            )
            stream.download(
                output_path=str(out_dir),
                filename=f"{video_id}.mp4",
            )
            filepath = out_dir / f"{video_id}.mp4"
        else:
            # Fallback: adaptive streams — need ffmpeg mux
            filepath = _download_adaptive(yt, video_id, out_dir)

        if filepath is None or not filepath.exists():
            logger.error("Download produced no file for %s", url)
            return None

        meta = {
            "url": url,
            "youtube_id": video_id,
            "title": yt.title,
            "uploader": yt.author,
            "upload_date": (
                yt.publish_date.strftime("%Y%m%d") if yt.publish_date else None
            ),
            "duration_sec": yt.length,
            "filepath": filepath,
        }
        logger.info(
            "Downloaded: %s  (%s, %ss)",
            meta["title"],
            video_id,
            meta["duration_sec"],
        )
        return meta

    except Exception:
        logger.exception("Failed to download %s", url)
        return None


# ── Helpers ──────────────────────────────────────────────


def _download_adaptive(
    yt: YouTube, video_id: str, out_dir: Path
) -> Path | None:
    """Download best adaptive video + audio and mux with ffmpeg.

    Returns the path to the muxed MP4, or None on failure.
    """
    # Pick best video stream ≤1080p
    video_stream = (
        yt.streams
        .filter(adaptive=True, file_extension="mp4", only_video=True)
        .order_by("resolution")
        .desc()
        .first()
    )
    # Pick best audio stream
    audio_stream = (
        yt.streams
        .filter(adaptive=True, only_audio=True)
        .order_by("abr")
        .desc()
        .first()
    )

    if not video_stream or not audio_stream:
        logger.error("No suitable adaptive streams found for %s", yt.title)
        return None

    if not shutil.which("ffmpeg"):
        logger.error(
            "ffmpeg required for adaptive stream muxing but not found on PATH"
        )
        return None

    logger.info(
        "Downloading (adaptive %s + audio): %s",
        video_stream.resolution,
        yt.title,
    )

    # Download to temp files
    video_tmp = out_dir / f"{video_id}_video_tmp.mp4"
    audio_ext = audio_stream.subtype or "m4a"
    audio_tmp = out_dir / f"{video_id}_audio_tmp.{audio_ext}"
    final_path = out_dir / f"{video_id}.mp4"

    try:
        video_stream.download(
            output_path=str(out_dir),
            filename=video_tmp.name,
        )
        audio_stream.download(
            output_path=str(out_dir),
            filename=audio_tmp.name,
        )

        # Mux with ffmpeg
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "warning",
            "-y",
            "-i", str(video_tmp),
            "-i", str(audio_tmp),
            "-c", "copy",
            str(final_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return final_path

    except Exception:
        logger.exception("Adaptive download/mux failed for %s", video_id)
        return None

    finally:
        # Clean up temp files
        video_tmp.unlink(missing_ok=True)
        audio_tmp.unlink(missing_ok=True)
