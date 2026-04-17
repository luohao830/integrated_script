import json
from pathlib import Path
from typing import Tuple

from integrated_script.config.settings import ConfigManager
from integrated_script.processors.yolo_processor import YOLOProcessor


def _build_processor(tmp_path: Path) -> YOLOProcessor:
    config = ConfigManager(config_file=tmp_path / "config.json", auto_save=False)
    config.set("paths.temp_dir", str(tmp_path / "temp"))
    config.set("paths.log_dir", str(tmp_path / "logs"))
    return YOLOProcessor(config=config)


def _create_basic_yolo_dataset(dataset: Path) -> Tuple[Path, Path]:
    images = dataset / "images"
    labels = dataset / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    return images, labels


def _create_basic_ctds_dataset(dataset: Path) -> None:
    obj_train_data = dataset / "obj_train_data"
    obj_train_data.mkdir(parents=True)

    (dataset / "obj.names").write_text("car\n", encoding="utf-8")
    (obj_train_data / "sample.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (obj_train_data / "sample.jpg").write_text("img", encoding="utf-8")


def _create_basic_ctds_segmentation_dataset(dataset: Path) -> None:
    obj_train_data = dataset / "obj_train_data"
    obj_train_data.mkdir(parents=True)

    (dataset / "obj.names").write_text("car\n", encoding="utf-8")
    (obj_train_data / "sample.txt").write_text(
        "0 0.1 0.1 0.2 0.2 0.3 0.3\n", encoding="utf-8"
    )
    (obj_train_data / "sample.jpg").write_text("img", encoding="utf-8")


def test_get_dataset_statistics_accepts_labels_subdir_path(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    images, labels = _create_basic_yolo_dataset(dataset)

    (images / "a.jpg").write_text("img", encoding="utf-8")
    (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (dataset / "classes.txt").write_text("cat\n", encoding="utf-8")

    processor = _build_processor(tmp_path)
    result = processor.get_dataset_statistics(str(labels))

    stats = result["statistics"]
    assert stats["dataset_path"] == str(dataset)
    assert stats["original_path"] == str(labels)
    assert stats["is_valid"] is True
    assert stats["has_classes_file"] is True
    assert stats["num_classes"] == 1
    assert stats["class_names"] == ["cat"]


def test_detect_yolo_dataset_type_returns_detection_for_bbox_labels(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    images, labels = _create_basic_yolo_dataset(dataset)

    (images / "a.jpg").write_text("img", encoding="utf-8")
    (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    processor = _build_processor(tmp_path)
    result = processor.detect_yolo_dataset_type(str(dataset))

    assert result["detected_type"] == "detection"
    assert result["statistics"]["total_files"] == 1
    assert result["statistics"]["detection_files"] == 1
    assert result["statistics"]["segmentation_files"] == 0


def test_detect_yolo_dataset_type_returns_segmentation_for_polygon_labels(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    images, labels = _create_basic_yolo_dataset(dataset)

    (images / "a.jpg").write_text("img", encoding="utf-8")
    (labels / "a.txt").write_text("0 0.1 0.1 0.2 0.2 0.3 0.3\n", encoding="utf-8")

    processor = _build_processor(tmp_path)
    result = processor.detect_yolo_dataset_type(str(dataset))

    assert result["detected_type"] == "segmentation"
    assert result["statistics"]["detection_files"] == 0
    assert result["statistics"]["segmentation_files"] == 1


def test_detect_xlabel_dataset_type_reports_segmentation_like_shapes(tmp_path: Path) -> None:
    source = tmp_path / "xlabel"
    source.mkdir(parents=True)

    payload = {
        "shapes": [
            {
                "label": "car",
                "shape_type": "polygon",
                "points": [[1, 1], [2, 1], [2, 2], [1, 2]],
            }
        ]
    }
    (source / "sample.json").write_text(json.dumps(payload), encoding="utf-8")

    processor = _build_processor(tmp_path)
    result = processor.detect_xlabel_dataset_type(str(source))

    assert result["detected_type"] == "segmentation"
    assert result["statistics"]["segmentation_like"] == 1
    assert result["statistics"]["detection_like"] == 0


def test_process_ctds_dataset_returns_pre_detection_stage_before_confirmation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "ctds"
    _create_basic_ctds_dataset(source)

    processor = _build_processor(tmp_path)
    result = processor.process_ctds_dataset(str(source), output_name="demo")

    assert result["success"] is True
    assert result["stage"] == "pre_detection"
    assert result["pre_detection_result"]["success"] is True
    assert result["pre_detection_result"]["dataset_type"] == "detection"
    assert not (tmp_path / "demo").exists()


def test_continue_ctds_processing_produces_output_after_confirmation(tmp_path: Path) -> None:
    source = tmp_path / "ctds"
    _create_basic_ctds_dataset(source)

    processor = _build_processor(tmp_path)
    pre_result = processor.process_ctds_dataset(str(source), output_name="demo")
    result = processor.continue_ctds_processing(pre_result, confirmed_type="detection")

    output_path = Path(result["output_path"])

    assert result["success"] is True
    assert result["detected_dataset_type"] == "detection"
    assert result["statistics"]["final_count"] == 1
    assert output_path.exists()
    assert (output_path / "images").exists()
    assert (output_path / "labels").exists()
    assert len(list((output_path / "images").glob("*.jpg"))) == 1
    assert len(list((output_path / "labels").glob("*.txt"))) == 1
    assert (output_path / "classes.txt").exists()
    assert not (source / "obj_train_data" / "sample.jpg").exists()
    assert not (source / "obj_train_data" / "sample.txt").exists()




def test_continue_ctds_processing_supports_segmentation_confirmation(tmp_path: Path) -> None:
    source = tmp_path / "ctds-seg"
    _create_basic_ctds_segmentation_dataset(source)

    processor = _build_processor(tmp_path)
    pre_result = processor.process_ctds_dataset(str(source), output_name="demo-seg")
    assert pre_result["stage"] == "pre_detection"

    result = processor.continue_ctds_processing(pre_result, confirmed_type="segmentation")
    output_path = Path(result["output_path"])

    assert result["success"] is True
    assert result["detected_dataset_type"] == "segmentation"
    assert result["statistics"]["final_count"] == 1
    assert output_path.exists()
    assert len(list((output_path / "images").glob("*.jpg"))) == 1
    assert len(list((output_path / "labels").glob("*.txt"))) == 1


def test_clean_unmatched_files_dry_run_collects_orphans_without_deleting(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    images, labels = _create_basic_yolo_dataset(dataset)

    orphan_image = images / "only_image.jpg"
    matched_image = images / "match.jpg"
    orphan_label = labels / "only_label.txt"
    matched_label = labels / "match.txt"

    orphan_image.write_text("img", encoding="utf-8")
    matched_image.write_text("img", encoding="utf-8")
    orphan_label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    matched_label.write_text("0 0.4 0.4 0.2 0.2\n", encoding="utf-8")

    processor = _build_processor(tmp_path)
    result = processor.clean_unmatched_files(str(dataset), dry_run=True)

    dry_run_deleted_images = {Path(path).name for path in result["deleted_files"]["orphaned_images"]}
    dry_run_deleted_labels = {Path(path).name for path in result["deleted_files"]["orphaned_labels"]}

    assert result["success"] is True
    assert "only_image.jpg" in dry_run_deleted_images
    assert "only_label.txt" in dry_run_deleted_labels
    assert orphan_image.exists()
    assert orphan_label.exists()


def test_clean_unmatched_files_deletes_orphans_when_not_dry_run(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    images, labels = _create_basic_yolo_dataset(dataset)

    orphan_image = images / "only_image.jpg"
    matched_image = images / "match.jpg"
    orphan_label = labels / "only_label.txt"
    matched_label = labels / "match.txt"

    orphan_image.write_text("img", encoding="utf-8")
    matched_image.write_text("img", encoding="utf-8")
    orphan_label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    matched_label.write_text("0 0.4 0.4 0.2 0.2\n", encoding="utf-8")

    processor = _build_processor(tmp_path)
    result = processor.clean_unmatched_files(str(dataset), dry_run=False)

    assert result["success"] is True
    assert not orphan_image.exists()
    assert not orphan_label.exists()
    assert matched_image.exists()
    assert matched_label.exists()
