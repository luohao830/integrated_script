from pathlib import Path

import pytest

from integrated_script.config.settings import ConfigManager
from integrated_script.processors.yolo.helpers import (
    build_label_mapping,
    format_duration,
)
from integrated_script.processors.yolo_processor import YOLOProcessor


def _build_processor(tmp_path: Path) -> YOLOProcessor:
    config = ConfigManager(config_file=tmp_path / "config.json", auto_save=False)
    config.set("paths.temp_dir", str(tmp_path / "temp"))
    config.set("paths.log_dir", str(tmp_path / "logs"))
    return YOLOProcessor(config=config)


@pytest.mark.unit
def test_format_duration_outputs_readable_units() -> None:
    assert format_duration(12.3) == "12.3秒"
    assert format_duration(125.0) == "2分5.0秒"
    assert format_duration(3661.2) == "1小时1分1.2秒"


@pytest.mark.unit
def test_build_label_mapping_skips_classes_file(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True)

    classes_file = labels_dir / "classes.txt"
    label_a = labels_dir / "a.txt"
    label_b = labels_dir / "b.txt"

    classes_file.write_text("cat\ndog\n", encoding="utf-8")
    label_a.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    label_b.write_text("1 0.4 0.4 0.2 0.2\n", encoding="utf-8")

    mapping = build_label_mapping([classes_file, label_a, label_b], "classes.txt")

    assert set(mapping.keys()) == {"a", "b"}
    assert mapping["a"] == label_a
    assert mapping["b"] == label_b


@pytest.mark.unit
def test_processor_facade_format_duration_matches_helper(tmp_path: Path) -> None:
    processor = _build_processor(tmp_path)

    assert processor._format_duration(12.3) == format_duration(12.3)
    assert processor._format_duration(125.0) == format_duration(125.0)
    assert processor._format_duration(3661.2) == format_duration(3661.2)


@pytest.mark.unit
def test_processor_facade_build_label_mapping_matches_helper(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True)

    classes_file = labels_dir / "classes.txt"
    label_a = labels_dir / "a.txt"
    label_b = labels_dir / "b.txt"

    classes_file.write_text("cat\ndog\n", encoding="utf-8")
    label_a.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    label_b.write_text("1 0.4 0.4 0.2 0.2\n", encoding="utf-8")

    processor = _build_processor(tmp_path)
    actual = processor._build_label_mapping([classes_file, label_a, label_b])
    expected = build_label_mapping([classes_file, label_a, label_b], "classes.txt")

    assert actual == expected
