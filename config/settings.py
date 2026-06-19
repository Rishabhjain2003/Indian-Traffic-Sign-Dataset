"""
Centralised configuration for the Indian Traffic Sign Detection Pipeline.

Every setting is a dataclass field that reads from an environment variable
at instantiation time, falling back to a sensible default.  Import the
module-level ``settings`` singleton wherever you need config values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _output_root() -> Path:
    """Return (and create) the top-level output directory."""
    root = Path("output")
    root.mkdir(parents=True, exist_ok=True)
    return root


@dataclass
class Settings:
    """Pipeline-wide configuration, overridable via environment variables."""

    # ── Database ─────────────────────────────────────────
    database_url: str = field(
        default_factory=lambda: os.environ.get(
            "DATABASE_URL",
            "postgresql://traffic:traffic@localhost:5432/indian_traffic_signs",
        )
    )

    # ── YOLO Model ───────────────────────────────────────
    yolo_weights: Path = field(
        default_factory=lambda: Path(
            os.environ.get("YOLO_WEIGHTS", "best.pt")
        )
    )
    yolo_device: str = field(
        default_factory=lambda: os.environ.get("YOLO_DEVICE", "cpu")
    )
    yolo_conf_threshold: float = field(
        default_factory=lambda: float(
            os.environ.get("YOLO_CONF_THRESHOLD", "0.35")
        )
    )
    yolo_iou_threshold: float = field(
        default_factory=lambda: float(
            os.environ.get("YOLO_IOU_THRESHOLD", "0.45")
        )
    )
    yolo_img_size: int = field(
        default_factory=lambda: int(os.environ.get("YOLO_IMG_SIZE", "640"))
    )
    yolo_batch_size: int = field(
        default_factory=lambda: int(os.environ.get("YOLO_BATCH_SIZE", "16"))
    )

    # ── Frame Extraction ────────────────────────────────
    fps: int = field(
        default_factory=lambda: int(os.environ.get("FPS", "30"))
    )
    frame_stride: int = field(
        default_factory=lambda: int(os.environ.get("FRAME_STRIDE", "1"))
    )

    # ── YouTube Search ───────────────────────────────────
    keywords_file: Path = field(
        default_factory=lambda: Path(
            os.environ.get("KEYWORDS_FILE", "combinatorial_keywords.txt")
        )
    )
    max_results_per_keyword: int = field(
        default_factory=lambda: int(
            os.environ.get("MAX_RESULTS_PER_KEYWORD", "5")
        )
    )
    search_delay: float = field(
        default_factory=lambda: float(
            os.environ.get("SEARCH_DELAY", "1.0")
        )
    )

    # ── Output ───────────────────────────────────────────
    save_annotated_frames: bool = field(
        default_factory=lambda: os.environ.get(
            "SAVE_ANNOTATED_FRAMES", "true"
        ).lower()
        in ("true", "1", "yes")
    )
    delete_video_after_processing: bool = field(
        default_factory=lambda: os.environ.get(
            "DELETE_VIDEO_AFTER_PROCESSING", "true"
        ).lower()
        in ("true", "1", "yes")
    )
    num_workers: int = field(
        default_factory=lambda: int(os.environ.get("NUM_WORKERS", "2"))
    )

    # ── Directory helpers (auto-create on access) ───────

    @property
    def videos_dir(self) -> Path:
        """Directory where downloaded videos are stored."""
        d = _output_root() / "videos"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def frames_dir(self) -> Path:
        """Directory where extracted frames are stored."""
        d = _output_root() / "frames"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def detections_dir(self) -> Path:
        """Directory where annotated detection frames are stored."""
        d = _output_root() / "detections"
        d.mkdir(parents=True, exist_ok=True)
        return d


# ── Module-level singleton ───────────────────────────────
settings = Settings()
