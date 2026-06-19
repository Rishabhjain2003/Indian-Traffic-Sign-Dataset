"""
YOLO Traffic Sign Detector — run YOLOv8 inference on extracted frames.

Loads a trained ``.pt`` weights file via the ``ultralytics`` library and
processes frames in configurable batches.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from config.settings import Settings

logger = logging.getLogger(__name__)


class TrafficSignDetector:
    """Wraps a YOLOv8 model for batch traffic sign detection."""

    def __init__(self, settings: Settings) -> None:
        weights = settings.yolo_weights
        if not weights.exists():
            raise FileNotFoundError(
                f"YOLO weights not found at {weights}. "
                "Train a model first — see TRAINING.md for instructions."
            )

        logger.info("Loading YOLO model from %s …", weights)
        self.model = YOLO(str(weights))
        self.settings = settings
        logger.info(
            "Model loaded — %d classes, device=%s",
            len(self.model.names),
            settings.yolo_device,
        )

    # ── Public API ───────────────────────────────────────

    def detect_frames(
        self,
        frame_infos: list[dict],
        video_id: str,
    ) -> list[dict]:
        """Run detection on a list of frame info dicts.

        Parameters
        ----------
        frame_infos:
            List of dicts each containing ``frame_number``, ``frame_path``,
            and ``timestamp_sec``.
        video_id:
            YouTube video ID (used for organising annotated output).

        Returns
        -------
        list[dict]
            Only frames with at least one detection.  Each dict is the
            original frame info dict enriched with a ``detections`` key
            containing a list of detection dicts.
        """
        s = self.settings
        batch_size = s.yolo_batch_size
        detected: list[dict] = []

        total = len(frame_infos)
        logger.info(
            "Running detection on %d frames (batch=%d, conf=%.2f) …",
            total,
            batch_size,
            s.yolo_conf_threshold,
        )

        # Prepare annotated output dir if needed
        ann_dir: Path | None = None
        if s.save_annotated_frames:
            ann_dir = s.detections_dir / video_id
            ann_dir.mkdir(parents=True, exist_ok=True)

        for batch_start in range(0, total, batch_size):
            batch = frame_infos[batch_start : batch_start + batch_size]
            paths = [str(fi["frame_path"]) for fi in batch]

            results = self.model(
                paths,
                conf=s.yolo_conf_threshold,
                iou=s.yolo_iou_threshold,
                imgsz=s.yolo_img_size,
                device=s.yolo_device,
                verbose=False,
            )

            for fi, result in zip(batch, results):
                dets = self._parse_result(result)
                if not dets:
                    continue

                fi_with_dets = {**fi, "detections": dets}
                detected.append(fi_with_dets)

                # Save annotated frame
                if ann_dir is not None:
                    self._save_annotated(result, fi["frame_path"], ann_dir)

            processed = min(batch_start + batch_size, total)
            logger.info(
                "  Batch %d–%d / %d  (detections so far: %d)",
                batch_start + 1,
                processed,
                total,
                len(detected),
            )

        logger.info(
            "Detection complete — %d / %d frames contain traffic signs",
            len(detected),
            total,
        )
        return detected

    # ── Internal helpers ─────────────────────────────────

    def _parse_result(self, result: Any) -> list[dict]:
        """Extract detection dicts from a single YOLO result."""
        detections: list[dict] = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": self.model.names.get(cls_id, f"class_{cls_id}"),
                    "confidence": round(float(box.conf[0].item()), 4),
                    "bbox": [round(float(c), 2) for c in box.xyxy[0].tolist()],
                }
            )
        return detections

    def _save_annotated(
        self, result: Any, frame_path: Path, ann_dir: Path
    ) -> None:
        """Draw bounding boxes on the frame and save to the detections dir."""
        try:
            annotated = result.plot()  # Returns a numpy BGR array
            out_path = ann_dir / Path(frame_path).name
            cv2.imwrite(str(out_path), annotated)
        except Exception:
            logger.debug(
                "Could not save annotated frame %s", frame_path, exc_info=True
            )
