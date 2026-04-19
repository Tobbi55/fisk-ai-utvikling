# pylint: skip-file
# mypy: ignore-errors

"""Export reviewed detections from a processed video to a YOLO dataset.

This tool reads sidecar files written next to a processed video:
- *.mapping.json
- *.predictions.json
- *.corrections.json (created/updated by the review UI)

It then exports frames (from the processed video) and labels (YOLO txt) so the
reviewed deletions become negative examples (empty label files).

Default behavior exports ONLY frames that were corrected (appear in corrections).
Use --include-uncorrected to also export model predictions for unreviewed frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import cv2


def _stable_split(key: str, val_ratio: float) -> str:
    if val_ratio <= 0:
        return "train"
    h = hashlib.md5(key.encode("utf-8"), usedforsecurity=False).hexdigest()
    bucket = int(h[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_ratio else "train"


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _to_yolo_line(
    cls_id: int,
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    width: int,
    height: int,
) -> str | None:
    # Ensure sane box
    xmin = max(0, min(width - 1, int(xmin)))
    xmax = max(0, min(width - 1, int(xmax)))
    ymin = max(0, min(height - 1, int(ymin)))
    ymax = max(0, min(height - 1, int(ymax)))

    if xmax <= xmin or ymax <= ymin:
        return None

    x_center = ((xmin + xmax) / 2.0) / float(width)
    y_center = ((ymin + ymax) / 2.0) / float(height)
    bw = (xmax - xmin) / float(width)
    bh = (ymax - ymin) / float(height)

    x_center = _clamp01(x_center)
    y_center = _clamp01(y_center)
    bw = _clamp01(bw)
    bh = _clamp01(bh)

    return f"{cls_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export reviewed corrections from a processed video to YOLO dataset"
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to *_processed.mp4",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("datasets/review_exports"),
        help="Output dataset root (a run folder will be created inside)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="Deterministic val split ratio (0 disables val)",
    )
    parser.add_argument(
        "--include-uncorrected",
        action="store_true",
        help="Also export frames that were not corrected, using model predictions",
    )
    parser.add_argument(
        "--class-name",
        type=str,
        default="fish",
        help="Single class name used for all boxes",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Limit number of exported frames (0 = no limit)",
    )
    parser.add_argument(
        "--image-format",
        choices=["jpg", "png"],
        default="jpg",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPG quality (0-100)",
    )

    args = parser.parse_args()

    video_path: Path = args.video
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    mapping_path = video_path.with_suffix(".mapping.json")
    predictions_path = video_path.with_suffix(".predictions.json")
    corrections_path = video_path.with_suffix(".corrections.json")

    if not mapping_path.exists() or not predictions_path.exists():
        raise SystemExit(
            "Missing sidecar files. Re-process the video with the current app version: "
            f"{mapping_path.name} / {predictions_path.name}"
        )

    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    output_to_original = [int(x) for x in mapping.get("output_to_original", [])]

    preds = json.loads(predictions_path.read_text(encoding="utf-8"))
    pred_map = preds.get("predictions", {})  # original_frame -> list[box]

    corrected_map: dict[str, list[dict]] = {}
    if corrections_path.exists():
        corr = json.loads(corrections_path.read_text(encoding="utf-8"))
        corrected_map = corr.get("corrected", {})

    if not corrected_map and not args.include_uncorrected:
        raise SystemExit(
            "No corrections found. Open the video in Review first to delete false boxes, "
            "or pass --include-uncorrected to export raw predictions."
        )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out / f"{video_path.stem}_{run_id}"

    img_train = run_dir / "images" / "train"
    lbl_train = run_dir / "labels" / "train"
    img_val = run_dir / "images" / "val"
    lbl_val = run_dir / "labels" / "val"

    img_train.mkdir(parents=True, exist_ok=True)
    lbl_train.mkdir(parents=True, exist_ok=True)
    if args.val_ratio > 0:
        img_val.mkdir(parents=True, exist_ok=True)
        lbl_val.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    exported = 0
    meta_path = run_dir / "meta.jsonl"

    def get_boxes_for_output_frame(out_idx: int) -> list[dict]:
        # If corrected => use that, can be empty (negative)
        key = str(out_idx)
        if key in corrected_map:
            return list(corrected_map[key])

        if not args.include_uncorrected:
            return []

        if out_idx < 0 or out_idx >= len(output_to_original):
            return []
        orig = output_to_original[out_idx]
        return list(pred_map.get(str(orig), []))

    def should_export(out_idx: int, boxes: list[dict]) -> bool:
        if str(out_idx) in corrected_map:
            return True
        return args.include_uncorrected and (len(boxes) > 0)

    with meta_path.open("w", encoding="utf-8") as meta_file:
        for out_idx in range(len(output_to_original)):
            boxes = get_boxes_for_output_frame(out_idx)
            if not should_export(out_idx, boxes):
                continue

            split = _stable_split(f"{video_path.name}:{out_idx}", args.val_ratio)
            img_dir = img_train if split == "train" else img_val
            lbl_dir = lbl_train if split == "train" else lbl_val

            cap.set(cv2.CAP_PROP_POS_FRAMES, float(out_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            height, width, _ = frame.shape

            fname = f"{out_idx:06d}.{args.image_format}"
            img_path = img_dir / fname
            lbl_path = lbl_dir / f"{out_idx:06d}.txt"

            # Write image
            if args.image_format == "jpg":
                cv2.imwrite(
                    str(img_path),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpg_quality)],
                )
            else:
                cv2.imwrite(str(img_path), frame)

            # Write label (empty file means negative)
            lines: list[str] = []
            for b in boxes:
                line = _to_yolo_line(
                    0,
                    int(b.get("xmin", 0)),
                    int(b.get("ymin", 0)),
                    int(b.get("xmax", 0)),
                    int(b.get("ymax", 0)),
                    width,
                    height,
                )
                if line is not None:
                    lines.append(line)

            lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

            meta_file.write(
                json.dumps(
                    {
                        "processed_video": video_path.name,
                        "output_frame": out_idx,
                        "original_frame": output_to_original[out_idx]
                        if out_idx < len(output_to_original)
                        else None,
                        "split": split,
                        "image": str(img_path.relative_to(run_dir)).replace(os.sep, "/"),
                        "label": str(lbl_path.relative_to(run_dir)).replace(os.sep, "/"),
                        "boxes": len(lines),
                        "corrected": str(out_idx) in corrected_map,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

            exported += 1
            if args.max_frames and exported >= args.max_frames:
                break

    cap.release()

    # Write a minimal dataset yaml
    yaml_path = run_dir / "data.yaml"
    yaml_text = "\n".join(
        [
            f"path: {run_dir.resolve()}",
            "train: images/train",
            "val: images/val" if args.val_ratio > 0 else "val: images/train",
            "nc: 1",
            f"names: [{args.class_name!r}]",
            "",
        ]
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")

    print(f"Exported {exported} frames to {run_dir}")
    print(f"Dataset yaml: {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
