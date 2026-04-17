from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from integrated_script.config.settings import ConfigManager
from integrated_script.processors import yolo_processor
from integrated_script.processors.yolo.ctds import get_project_name
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


@pytest.mark.unit
def test_get_project_name_reads_obj_names_when_manual_name_missing(
    tmp_path: Path,
) -> None:
    processor = _build_processor(tmp_path)
    obj_names = tmp_path / "obj.names"
    obj_names.write_text("car\ntruck\n", encoding="utf-8")

    assert get_project_name(processor, obj_names) == "car-truck"


@pytest.mark.unit
def test_processor_facade_process_ctds_dataset_delegates_to_internal_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = _build_processor(tmp_path)
    expected = {"success": True, "stage": "pre_detection"}

    captured: Dict[str, Any] = {}

    def fake_process_ctds_dataset_internal(
        self: YOLOProcessor,
        input_path: str,
        output_name: Optional[str],
        keep_empty_labels: bool,
    ) -> Dict[str, Any]:
        captured["self"] = self
        captured["input_path"] = input_path
        captured["output_name"] = output_name
        captured["keep_empty_labels"] = keep_empty_labels
        return expected

    monkeypatch.setattr(
        yolo_processor,
        "process_ctds_dataset_internal",
        fake_process_ctds_dataset_internal,
    )

    result = processor.process_ctds_dataset(
        "/tmp/ctds",
        output_name="demo",
        keep_empty_labels=True,
    )

    assert result == expected
    assert captured == {
        "self": processor,
        "input_path": "/tmp/ctds",
        "output_name": "demo",
        "keep_empty_labels": True,
    }


@pytest.mark.unit
def test_processor_facade_continue_ctds_processing_delegates_to_internal_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = _build_processor(tmp_path)
    pre_result: Dict[str, Any] = {"input_path": "/tmp/ctds"}
    expected = {"success": True, "detected_dataset_type": "detection"}

    captured: Dict[str, Any] = {}

    def fake_continue_ctds_processing_internal(
        self: YOLOProcessor,
        pre_result: Dict[str, Any],
        confirmed_type: str,
        keep_empty_labels: Optional[bool],
    ) -> Dict[str, Any]:
        captured["self"] = self
        captured["pre_result"] = pre_result
        captured["confirmed_type"] = confirmed_type
        captured["keep_empty_labels"] = keep_empty_labels
        return expected

    monkeypatch.setattr(
        yolo_processor,
        "continue_ctds_processing_internal",
        fake_continue_ctds_processing_internal,
    )

    result = processor.continue_ctds_processing(
        pre_result,
        confirmed_type="detection",
        keep_empty_labels=False,
    )

    assert result == expected
    assert captured == {
        "self": processor,
        "pre_result": pre_result,
        "confirmed_type": "detection",
        "keep_empty_labels": False,
    }


@pytest.mark.unit
def test_processor_facade_get_dataset_statistics_delegates_to_internal_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = _build_processor(tmp_path)
    expected = {"valid": True, "statistics": {"is_valid": True}}

    captured: Dict[str, Any] = {}

    def fake_get_dataset_statistics_internal(
        self: YOLOProcessor,
        dataset_path: str,
    ) -> Dict[str, Any]:
        captured["self"] = self
        captured["dataset_path"] = dataset_path
        return expected

    monkeypatch.setattr(
        yolo_processor,
        "get_dataset_statistics_internal",
        fake_get_dataset_statistics_internal,
    )

    result = processor.get_dataset_statistics("/tmp/yolo")

    assert result == expected
    assert captured == {
        "self": processor,
        "dataset_path": "/tmp/yolo",
    }


@pytest.mark.unit
def test_processor_facade_clean_unmatched_files_delegates_to_internal_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = _build_processor(tmp_path)
    expected = {
        "success": True,
        "dry_run": True,
        "statistics": {"total_deleted": 0},
    }

    captured: Dict[str, Any] = {}

    def fake_clean_unmatched_files_internal(
        self: YOLOProcessor,
        dataset_path: str,
        dry_run: bool,
    ) -> Dict[str, Any]:
        captured["self"] = self
        captured["dataset_path"] = dataset_path
        captured["dry_run"] = dry_run
        return expected

    monkeypatch.setattr(
        yolo_processor,
        "clean_unmatched_files_internal",
        fake_clean_unmatched_files_internal,
    )

    result = processor.clean_unmatched_files("/tmp/yolo", dry_run=True)

    assert result == expected
    assert captured == {
        "self": processor,
        "dataset_path": "/tmp/yolo",
        "dry_run": True,
    }


@pytest.mark.unit
def test_processor_facade_execute_ctds_processing_delegates_to_internal_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = _build_processor(tmp_path)
    expected = {"success": True, "finalized": True}

    captured: Dict[str, Any] = {}

    def fake_execute_ctds_processing_internal(
        self: YOLOProcessor,
        input_dir: Path,
        project_name: str,
        obj_names_path: Path,
        obj_train_data_path: Path,
        confirmed_type: str,
        pre_detection_result: Dict[str, Any],
        keep_empty_labels: bool,
    ) -> Dict[str, Any]:
        captured["self"] = self
        captured["input_dir"] = input_dir
        captured["project_name"] = project_name
        captured["obj_names_path"] = obj_names_path
        captured["obj_train_data_path"] = obj_train_data_path
        captured["confirmed_type"] = confirmed_type
        captured["pre_detection_result"] = pre_detection_result
        captured["keep_empty_labels"] = keep_empty_labels
        return expected

    monkeypatch.setattr(
        yolo_processor,
        "execute_ctds_processing_internal",
        fake_execute_ctds_processing_internal,
    )

    input_dir = tmp_path / "ctds"
    obj_names_path = input_dir / "obj.names"
    obj_train_data_path = input_dir / "obj_train_data"
    pre_detection_result = {"success": True, "dataset_type": "detection"}

    result = processor._execute_ctds_processing(
        input_dir=input_dir,
        project_name="demo",
        obj_names_path=obj_names_path,
        obj_train_data_path=obj_train_data_path,
        confirmed_type="detection",
        pre_detection_result=pre_detection_result,
        keep_empty_labels=True,
    )

    assert result == expected
    assert captured == {
        "self": processor,
        "input_dir": input_dir,
        "project_name": "demo",
        "obj_names_path": obj_names_path,
        "obj_train_data_path": obj_train_data_path,
        "confirmed_type": "detection",
        "pre_detection_result": pre_detection_result,
        "keep_empty_labels": True,
    }


@pytest.mark.unit
def test_processor_facade_get_project_name_delegates_to_internal_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = _build_processor(tmp_path)
    expected = "project-from-helper"

    captured: Dict[str, Any] = {}

    def fake_get_project_name(
        self: YOLOProcessor,
        obj_names_path: Path,
        manual_name: Optional[str],
    ) -> str:
        captured["self"] = self
        captured["obj_names_path"] = obj_names_path
        captured["manual_name"] = manual_name
        return expected

    monkeypatch.setattr(yolo_processor, "get_project_name", fake_get_project_name)

    obj_names_path = tmp_path / "obj.names"
    result = processor._get_project_name(obj_names_path, manual_name="demo")

    assert result == expected
    assert captured == {
        "self": processor,
        "obj_names_path": obj_names_path,
        "manual_name": "demo",
    }
