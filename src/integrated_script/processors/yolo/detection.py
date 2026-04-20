"""Dataset type detection internals for YOLO processor."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set


def list_xlabel_json_files_recursive(source_path: Path) -> List[Path]:
    """按历史语义递归收集 JSON，跳过 *_dataset 目录。"""
    json_files: List[Path] = []
    for root, dirs, files in os.walk(source_path):
        root_path = Path(root)
        if root_path.name.endswith("_dataset"):
            dirs[:] = []
            continue

        for filename in files:
            if filename.lower().endswith(".json"):
                json_files.append(root_path / filename)

    return json_files


def scan_xlabel_dataset_recursive(
    source_path: Path,
    warn_json_read_failed: Optional[Callable[[Path, Exception], None]] = None,
) -> Dict[str, Any]:
    """递归扫描 X-label JSON，一次遍历收集类别与形状统计。"""
    json_files = list_xlabel_json_files_recursive(source_path)

    classes: Set[str] = set()
    total_shapes = 0
    detection_like = 0
    segmentation_like = 0

    for json_path in json_files:
        try:
            with json_path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            for shape in data.get("shapes", []):
                label = shape.get("label")
                if label:
                    classes.add(label)

                points = shape.get("points", [])
                shape_type = (shape.get("shape_type") or "").lower()
                if not points:
                    continue

                total_shapes += 1

                if shape_type in {"rectangle", "rect", "box"}:
                    detection_like += 1
                    continue

                if shape_type in {"polygon", "polyline", "linestrip"}:
                    segmentation_like += 1
                    continue

                if len(points) >= 3:
                    if len(points) == 4:
                        xs = {point[0] for point in points}
                        ys = {point[1] for point in points}
                        if len(xs) == 2 and len(ys) == 2:
                            detection_like += 1
                            continue
                    segmentation_like += 1
                else:
                    detection_like += 1

        except Exception as error:  # noqa: BLE001
            if warn_json_read_failed is not None:
                warn_json_read_failed(json_path, error)

    return {
        "json_files": json_files,
        "classes": classes,
        "total_shapes": total_shapes,
        "detection_like": detection_like,
        "segmentation_like": segmentation_like,
    }
