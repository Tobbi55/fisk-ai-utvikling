"""Review window for deleting false-positive detections on processed videos."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.logger import get_logger
from app.widgets.error_dialog import ErrorDialog

logger = get_logger()


@dataclass(frozen=True)
class Box:
    """A single bounding box in pixel coordinates."""

    label: str
    confidence: float
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    def contains(self, x: float, y: float) -> bool:
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax

    def as_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": float(self.confidence),
            "xmin": int(self.xmin),
            "ymin": int(self.ymin),
            "xmax": int(self.xmax),
            "ymax": int(self.ymax),
        }


class FrameView(QLabel):
    """A QLabel that displays a frame and lets users click to delete boxes."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(640, 360)

        self._original_size: tuple[int, int] | None = None
        self._display_size: tuple[int, int] | None = None
        self._display_offset: tuple[int, int] = (0, 0)
        self._boxes: list[Box] = []
        self._on_box_clicked: callable[[Box], None] | None = None

    def set_on_box_clicked(self, callback: callable[[Box], None]) -> None:
        self._on_box_clicked = callback

    def set_frame(self, frame_bgr: Any, boxes: list[Box]) -> None:
        # Convert BGR -> RGB and into QPixmap
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape
        self._original_size = (w, h)
        self._boxes = boxes

        qimg = QImage(
            frame_rgb.data,
            w,
            h,
            frame_rgb.strides[0],
            QImage.Format.Format_RGB888,
        )
        base_pixmap = QPixmap.fromImage(qimg)

        # Scale to current label size while keeping aspect ratio
        target = self.size()
        scaled = base_pixmap.scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._display_size = (scaled.width(), scaled.height())
        self._display_offset = (
            max(0, int((target.width() - scaled.width()) / 2)),
            max(0, int((target.height() - scaled.height()) / 2)),
        )

        # Draw boxes on the scaled pixmap
        annotated = QPixmap(scaled)
        painter = QPainter(annotated)
        pen = QPen(Qt.GlobalColor.red)
        pen.setWidth(2)
        painter.setPen(pen)

        if self._original_size is not None:
            ow, oh = self._original_size
            sx = scaled.width() / float(ow)
            sy = scaled.height() / float(oh)

            for box in boxes:
                x1 = int(box.xmin * sx)
                y1 = int(box.ymin * sy)
                x2 = int(box.xmax * sx)
                y2 = int(box.ymax * sy)
                painter.drawRect(x1, y1, max(1, x2 - x1), max(1, y2 - y1))

        painter.end()

        self.setPixmap(annotated)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        # Re-render current pixmap on resize by forcing caller to set_frame again.
        # This keeps the implementation minimal.
        super().resizeEvent(event)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if (
            self._on_box_clicked is None
            or self._original_size is None
            or self._display_size is None
            or self.pixmap() is None
        ):
            return

        click_x = float(event.position().x())
        click_y = float(event.position().y())

        off_x, off_y = self._display_offset
        disp_w, disp_h = self._display_size

        # Click outside displayed image
        if click_x < off_x or click_y < off_y:
            return
        if click_x > off_x + disp_w or click_y > off_y + disp_h:
            return

        # Map back to original pixel coordinates
        ow, oh = self._original_size
        rel_x = (click_x - off_x) / float(disp_w)
        rel_y = (click_y - off_y) / float(disp_h)
        orig_x = rel_x * ow
        orig_y = rel_y * oh

        # Prefer the smallest box that contains the click (more precise)
        candidate: Box | None = None
        candidate_area: int | None = None
        for box in self._boxes:
            if not box.contains(orig_x, orig_y):
                continue
            area = (box.xmax - box.xmin) * (box.ymax - box.ymin)
            if candidate is None or (candidate_area is not None and area < candidate_area):
                candidate = box
                candidate_area = area

        if candidate is not None:
            self._on_box_clicked(candidate)


class ReviewWindow(QDialog):
    """Dialog for reviewing a processed video and deleting false boxes."""

    def __init__(self, processed_video_path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Review detections")
        self.setMinimumSize(900, 600)

        self.processed_video_path = processed_video_path
        self.mapping_path = processed_video_path.with_suffix(".mapping.json")
        self.predictions_path = processed_video_path.with_suffix(".predictions.json")
        self.corrections_path = processed_video_path.with_suffix(".corrections.json")

        self.output_to_original: list[int] = []
        self.predictions_by_original: dict[int, list[Box]] = {}
        self.corrected_by_output: dict[int, list[Box]] = {}

        self.frames_with_boxes: list[int] = []
        self.current_output_frame: int = 0

        self.cap = cv2.VideoCapture(str(processed_video_path))
        if not self.cap.isOpened():
            ErrorDialog(f"Could not open video: {processed_video_path}", parent=self).exec()
            return

        if not self.mapping_path.exists() or not self.predictions_path.exists():
            ErrorDialog(
                "Missing review files (.mapping.json / .predictions.json). Please re-process the video using this version of the app.",
                parent=self,
            ).exec()
            return

        try:
            self._load_sidecars()
        except Exception as err:  # pylint: disable=broad-except
            ErrorDialog(f"Failed to load review data: {err}", parent=self).exec()
            return

        self._load_existing_corrections()
        self._rebuild_frames_with_boxes()

        self.root_layout = QVBoxLayout()

        self.info_label = QLabel("")
        self.root_layout.addWidget(self.info_label)

        self.frame_view = FrameView()
        self.frame_view.set_on_box_clicked(self._delete_box)
        self.root_layout.addWidget(self.frame_view)

        controls = QHBoxLayout()
        self.prev_btn = QPushButton("Prev")
        self.next_btn = QPushButton("Next")
        self.close_btn = QPushButton("Close")

        self.prev_btn.clicked.connect(self.prev_frame)
        self.next_btn.clicked.connect(self.next_frame)
        self.close_btn.clicked.connect(self.accept)

        controls.addWidget(self.prev_btn)
        controls.addWidget(self.next_btn)
        controls.addStretch()
        controls.addWidget(self.close_btn)
        self.root_layout.addLayout(controls)

        self.setLayout(self.root_layout)

        if self.frames_with_boxes:
            self.current_output_frame = self.frames_with_boxes[0]
        else:
            self.current_output_frame = 0

        self._render_current()

    def _load_sidecars(self) -> None:
        mapping_data = json.loads(self.mapping_path.read_text(encoding="utf-8"))
        self.output_to_original = [int(x) for x in mapping_data.get("output_to_original", [])]

        preds_data = json.loads(self.predictions_path.read_text(encoding="utf-8"))
        pred_map = preds_data.get("predictions", {})

        by_original: dict[int, list[Box]] = {}
        for frame_str, boxes in pred_map.items():
            frame = int(frame_str)
            by_original[frame] = [
                Box(
                    label=str(b.get("label", "fish")),
                    confidence=float(b.get("confidence", 0.0)),
                    xmin=int(b.get("xmin", 0)),
                    ymin=int(b.get("ymin", 0)),
                    xmax=int(b.get("xmax", 0)),
                    ymax=int(b.get("ymax", 0)),
                )
                for b in boxes
            ]
        self.predictions_by_original = by_original

    def _load_existing_corrections(self) -> None:
        if not self.corrections_path.exists():
            self.corrected_by_output = {}
            return

        try:
            data = json.loads(self.corrections_path.read_text(encoding="utf-8"))
            corrected = data.get("corrected", {})
            out: dict[int, list[Box]] = {}
            for out_idx_str, boxes in corrected.items():
                out_idx = int(out_idx_str)
                out[out_idx] = [
                    Box(
                        label=str(b.get("label", "fish")),
                        confidence=float(b.get("confidence", 0.0)),
                        xmin=int(b.get("xmin", 0)),
                        ymin=int(b.get("ymin", 0)),
                        xmax=int(b.get("xmax", 0)),
                        ymax=int(b.get("ymax", 0)),
                    )
                    for b in boxes
                ]
            self.corrected_by_output = out
        except Exception as err:  # pylint: disable=broad-except
            logger.warning("Failed to read corrections file %s: %s", self.corrections_path, err)
            self.corrected_by_output = {}

    def _save_corrections(self) -> None:
        payload = {
            "version": 1,
            "processed_video": self.processed_video_path.name,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "corrected": {
                str(k): [b.as_json() for b in v] for k, v in sorted(self.corrected_by_output.items())
            },
        }
        self.corrections_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _rebuild_frames_with_boxes(self) -> None:
        frames: list[int] = []
        for out_idx, orig in enumerate(self.output_to_original):
            if out_idx in self.corrected_by_output:
                if len(self.corrected_by_output[out_idx]) > 0:
                    frames.append(out_idx)
                continue

            if len(self.predictions_by_original.get(int(orig), [])) > 0:
                frames.append(out_idx)
        self.frames_with_boxes = frames

    def _get_boxes_for_output_frame(self, out_idx: int) -> list[Box]:
        if out_idx in self.corrected_by_output:
            return list(self.corrected_by_output[out_idx])
        if out_idx < 0 or out_idx >= len(self.output_to_original):
            return []
        orig = self.output_to_original[out_idx]
        return list(self.predictions_by_original.get(int(orig), []))

    def _render_current(self) -> None:
        if not self.output_to_original:
            self.info_label.setText("No mapping data")
            return

        out_idx = self.current_output_frame
        out_idx = max(0, min(out_idx, len(self.output_to_original) - 1))
        self.current_output_frame = out_idx

        # Seek and read frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, float(out_idx))
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.info_label.setText(f"Failed to read frame {out_idx}")
            return

        orig = self.output_to_original[out_idx]
        boxes = self._get_boxes_for_output_frame(out_idx)

        self.info_label.setText(
            f"{self.processed_video_path.name} | output frame {out_idx + 1}/{len(self.output_to_original)} | original frame {orig} | boxes {len(boxes)} | click box to delete"
        )

        self.frame_view.set_frame(frame, boxes)

        self.prev_btn.setEnabled(bool(self.frames_with_boxes) and out_idx != self.frames_with_boxes[0])
        self.next_btn.setEnabled(bool(self.frames_with_boxes) and out_idx != self.frames_with_boxes[-1])

    def _delete_box(self, box: Box) -> None:
        out_idx = self.current_output_frame
        boxes = self._get_boxes_for_output_frame(out_idx)

        # Remove one matching box
        removed = False
        new_boxes: list[Box] = []
        for b in boxes:
            if not removed and b == box:
                removed = True
                continue
            new_boxes.append(b)

        # Mark this frame as corrected (even if empty => 'no fish')
        self.corrected_by_output[out_idx] = new_boxes
        self._save_corrections()

        self._rebuild_frames_with_boxes()
        if self.frames_with_boxes:
            # If current frame has no boxes left, jump to next available
            if out_idx not in self.frames_with_boxes:
                next_candidates = [i for i in self.frames_with_boxes if i >= out_idx]
                self.current_output_frame = next_candidates[0] if next_candidates else self.frames_with_boxes[-1]

        self._render_current()

    def prev_frame(self) -> None:
        if not self.frames_with_boxes:
            return
        try:
            idx = self.frames_with_boxes.index(self.current_output_frame)
        except ValueError:
            idx = 0
        self.current_output_frame = self.frames_with_boxes[max(0, idx - 1)]
        self._render_current()

    def next_frame(self) -> None:
        if not self.frames_with_boxes:
            return
        try:
            idx = self.frames_with_boxes.index(self.current_output_frame)
        except ValueError:
            idx = -1
        self.current_output_frame = self.frames_with_boxes[min(len(self.frames_with_boxes) - 1, idx + 1)]
        self._render_current()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:  # pylint: disable=broad-except
            pass
        super().closeEvent(event)
