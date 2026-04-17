"""Shared helper functions for YOLO processor internals."""

from pathlib import Path
from typing import Dict, List


def build_label_mapping(label_files: List[Path], classes_file: str) -> Dict[str, Path]:
    """Build base-name to label-path index while skipping classes file."""
    label_mapping: Dict[str, Path] = {}
    for label_file in label_files:
        if label_file.name != classes_file:
            label_mapping[label_file.stem] = label_file
    return label_mapping


def format_duration(seconds: float) -> str:
    """Format duration in Chinese-readable units."""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    if seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}分{secs:.1f}秒"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}小时{minutes}分{secs:.1f}秒"
