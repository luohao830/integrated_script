from integrated_script.contracts.errors import NormalizedError
from integrated_script.workflows.yolo_workflow import YoloWorkflow


class _StubYoloProcessor:
    def process_ctds_dataset(
        self,
        _input_path: str,
        output_name=None,
        keep_empty_labels: bool = False,
    ):
        return {
            "success": True,
            "stage": "pre_detection",
            "project_name": output_name or "demo",
            "keep_empty_labels": keep_empty_labels,
        }

    def continue_ctds_processing(
        self,
        pre_result: dict,
        confirmed_type: str,
        keep_empty_labels=None,
    ):
        return {
            "success": True,
            "project_name": pre_result.get("project_name", "demo"),
            "detected_dataset_type": confirmed_type,
            "keep_empty_labels": keep_empty_labels,
        }

    def convert_yolo_to_ctds_dataset(
        self,
        dataset_path: str,
        output_path=None,
    ):
        return {
            "success": True,
            "dataset_path": dataset_path,
            "output_path": output_path or "/tmp/ctds",
            "statistics": {
                "total_labels": 1,
                "labels_copied": 1,
                "images_copied": 1,
                "missing_images": 0,
            },
            "missing_images": [],
        }

    def detect_yolo_dataset_type(self, dataset_path: str):
        return {
            "success": True,
            "dataset_path": dataset_path,
            "detected_type": "detection",
            "confidence": 0.95,
            "statistics": {
                "total_files": 3,
                "detection_files": 3,
                "segmentation_files": 0,
            },
        }

    def convert_yolo_to_xlabel(self, dataset_path: str, output_dir=None):
        return {
            "success": True,
            "dataset_path": dataset_path,
            "output_dir": output_dir or "/tmp/xlabel",
            "mode": "detection",
        }

    def convert_yolo_to_xlabel_segmentation(self, dataset_path: str, output_dir=None):
        return {
            "success": True,
            "dataset_path": dataset_path,
            "output_dir": output_dir or "/tmp/xlabel-seg",
            "mode": "segmentation",
        }

    def detect_xlabel_dataset_type(self, source_dir: str):
        return {
            "success": True,
            "source_dir": source_dir,
            "detected_type": "segmentation",
            "confidence": 0.9,
            "statistics": {
                "total_shapes": 6,
                "detection_like": 1,
                "segmentation_like": 5,
            },
        }

    def detect_xlabel_classes(self, _source_dir: str):
        return {"cat", "dog"}

    def detect_xlabel_segmentation_classes(self, _source_dir: str):
        return {"road", "car"}

    def convert_xlabel_to_yolo(
        self,
        source_dir: str,
        output_dir=None,
        class_order=None,
    ):
        return {
            "success": True,
            "source_dir": source_dir,
            "output_dir": output_dir or "/tmp/yolo",
            "class_order": class_order or ["cat", "dog"],
        }

    def convert_xlabel_to_yolo_segmentation(
        self,
        source_dir: str,
        output_dir=None,
        class_order=None,
    ):
        return {
            "success": True,
            "source_dir": source_dir,
            "output_dir": output_dir or "/tmp/yolo-seg",
            "class_order": class_order or ["road", "car"],
        }

    def get_dataset_statistics(self, dataset_path: str):
        return {
            "success": True,
            "dataset_path": dataset_path,
            "statistics": {
                "is_valid": True,
                "orphaned_images": 0,
                "orphaned_labels": 0,
            },
        }

    def clean_unmatched_files(self, dataset_path: str, dry_run: bool = False):
        return {
            "success": True,
            "dataset_path": dataset_path,
            "dry_run": dry_run,
            "deleted_files": {
                "orphaned_images": [],
                "orphaned_labels": [],
                "invalid_labels": [],
                "empty_labels": [],
            },
            "statistics": {
                "total_deleted": 0,
                "deleted_images": 0,
                "deleted_labels": 0,
            },
        }

    def _validate_classes_consistency(self, dataset_paths):
        return {
            "success": True,
            "consistent": True,
            "classes": ["cat", "dog"],
            "dataset_paths": dataset_paths,
        }

    def _generate_output_name(self, classes, dataset_paths):
        return f"merged-{'-'.join(classes)}-{len(dataset_paths)}"

    def merge_datasets(
        self,
        dataset_paths,
        output_path: str,
        output_name=None,
        image_prefix=None,
    ):
        return {
            "success": True,
            "dataset_paths": dataset_paths,
            "output_path": output_path,
            "output_name": output_name or "merged",
            "image_prefix": image_prefix,
            "total_images": 2,
            "total_labels": 2,
            "classes": ["cat", "dog"],
            "merged_datasets": len(dataset_paths),
        }

    def _collect_all_classes_info(self, dataset_paths):
        return [
            {"dataset_path": dataset_paths[0], "classes": ["cat"]},
            {"dataset_path": dataset_paths[1], "classes": ["dog"]},
        ]

    def _create_unified_class_mapping(self, _all_classes_info):
        return ["cat", "dog"], [{0: 0}, {0: 1}]

    def _generate_different_output_name(self, unified_classes, dataset_paths):
        return f"different-{'-'.join(unified_classes)}-{len(dataset_paths)}"

    def merge_different_type_datasets(
        self,
        dataset_paths,
        output_path: str,
        output_name=None,
        image_prefix=None,
        dataset_order=None,
    ):
        return {
            "success": True,
            "dataset_paths": dataset_paths,
            "output_path": output_path,
            "output_name": output_name or "merged-diff",
            "image_prefix": image_prefix,
            "dataset_order": dataset_order,
            "total_images": 2,
            "total_labels": 2,
            "unified_classes": ["cat", "dog"],
            "merged_datasets": len(dataset_paths),
            "statistics": [],
        }


class _StubYoloProcessorRaises:
    def process_ctds_dataset(
        self,
        _input_path: str,
        output_name=None,
        keep_empty_labels: bool = False,
    ):
        del output_name, keep_empty_labels
        raise RuntimeError("boom")


class _StubYoloWorkflow(YoloWorkflow):
    def _normalize_error(self, _error: Exception) -> NormalizedError:
        return NormalizedError(
            code="INTERNAL_ERROR",
            message="内部处理失败",
            details={"exception_type": "RuntimeError"},
        )


def test_yolo_workflow_process_ctds_dataset_returns_legacy_dict() -> None:
    workflow = YoloWorkflow(_StubYoloProcessor())

    result = workflow.process_ctds_dataset("/tmp/input", output_name="my-proj")

    assert isinstance(result, dict)
    assert result["success"] is True
    assert result["stage"] == "pre_detection"
    assert result["project_name"] == "my-proj"


def test_yolo_workflow_continue_ctds_processing_returns_legacy_dict() -> None:
    workflow = YoloWorkflow(_StubYoloProcessor())

    result = workflow.continue_ctds_processing(
        pre_result={"project_name": "my-proj"},
        confirmed_type="detection",
        keep_empty_labels=False,
    )

    assert isinstance(result, dict)
    assert result["success"] is True
    assert result["project_name"] == "my-proj"
    assert result["detected_dataset_type"] == "detection"
    assert result["keep_empty_labels"] is False


def test_yolo_workflow_convert_yolo_to_ctds_dataset_returns_legacy_dict() -> None:
    workflow = YoloWorkflow(_StubYoloProcessor())

    result = workflow.convert_yolo_to_ctds_dataset(
        dataset_path="/tmp/yolo",
        output_path="/tmp/ctds-out",
    )

    assert isinstance(result, dict)
    assert result["success"] is True
    assert result["dataset_path"] == "/tmp/yolo"
    assert result["output_path"] == "/tmp/ctds-out"
    assert result["statistics"]["labels_copied"] == 1
    assert result["statistics"]["images_copied"] == 1


def test_yolo_workflow_convert_yolo_to_ctds_dataset_returns_legacy_failure_on_exception() -> (
    None
):
    class _StubYoloProcessorConvertRaises(_StubYoloProcessor):
        def convert_yolo_to_ctds_dataset(self, dataset_path: str, output_path=None):
            raise RuntimeError(f"convert failed: {dataset_path}, {output_path}")

    workflow = _StubYoloWorkflow(_StubYoloProcessorConvertRaises())

    result = workflow.convert_yolo_to_ctds_dataset(
        dataset_path="/tmp/yolo",
        output_path="/tmp/ctds-out",
    )

    assert isinstance(result, dict)
    assert result["success"] is False
    assert result["error_code"] == "INTERNAL_ERROR"
    assert result["error"] == "内部处理失败"
    assert result["error_details"]["exception_type"] == "RuntimeError"


def test_yolo_workflow_detect_yolo_dataset_type_returns_legacy_dict() -> None:
    workflow = YoloWorkflow(_StubYoloProcessor())

    result = workflow.detect_yolo_dataset_type("/tmp/yolo")

    assert result["success"] is True
    assert result["detected_type"] == "detection"
    assert result["confidence"] == 0.95


def test_yolo_workflow_convert_yolo_to_xlabel_returns_legacy_dict() -> None:
    workflow = YoloWorkflow(_StubYoloProcessor())

    result = workflow.convert_yolo_to_xlabel("/tmp/yolo", output_dir="/tmp/xlabel")

    assert result["success"] is True
    assert result["dataset_path"] == "/tmp/yolo"
    assert result["output_dir"] == "/tmp/xlabel"


def test_yolo_workflow_convert_yolo_to_xlabel_segmentation_returns_legacy_dict() -> (
    None
):
    workflow = YoloWorkflow(_StubYoloProcessor())

    result = workflow.convert_yolo_to_xlabel_segmentation(
        "/tmp/yolo",
        output_dir="/tmp/xlabel-seg",
    )

    assert result["success"] is True
    assert result["dataset_path"] == "/tmp/yolo"
    assert result["output_dir"] == "/tmp/xlabel-seg"


def test_yolo_workflow_detect_xlabel_dataset_type_returns_legacy_dict() -> None:
    workflow = YoloWorkflow(_StubYoloProcessor())

    result = workflow.detect_xlabel_dataset_type("/tmp/xlabel")

    assert result["success"] is True
    assert result["detected_type"] == "segmentation"
    assert result["statistics"]["segmentation_like"] == 5


def test_yolo_workflow_detect_xlabel_classes_returns_passthrough_set() -> None:
    workflow = YoloWorkflow(_StubYoloProcessor())

    result = workflow.detect_xlabel_classes("/tmp/xlabel")

    assert result == {"cat", "dog"}


def test_yolo_workflow_detect_xlabel_segmentation_classes_returns_passthrough_set() -> (
    None
):
    workflow = YoloWorkflow(_StubYoloProcessor())

    result = workflow.detect_xlabel_segmentation_classes("/tmp/xlabel")

    assert result == {"road", "car"}


def test_yolo_workflow_convert_xlabel_to_yolo_returns_legacy_dict() -> None:
    workflow = YoloWorkflow(_StubYoloProcessor())

    result = workflow.convert_xlabel_to_yolo(
        source_dir="/tmp/xlabel",
        output_dir="/tmp/yolo-out",
        class_order=["cat", "dog"],
    )

    assert result["success"] is True
    assert result["source_dir"] == "/tmp/xlabel"
    assert result["output_dir"] == "/tmp/yolo-out"


def test_yolo_workflow_convert_xlabel_to_yolo_segmentation_returns_legacy_dict() -> (
    None
):
    workflow = YoloWorkflow(_StubYoloProcessor())

    result = workflow.convert_xlabel_to_yolo_segmentation(
        source_dir="/tmp/xlabel",
        output_dir="/tmp/yolo-seg-out",
        class_order=["road", "car"],
    )

    assert result["success"] is True
    assert result["source_dir"] == "/tmp/xlabel"
    assert result["output_dir"] == "/tmp/yolo-seg-out"


def test_yolo_workflow_get_dataset_statistics_returns_legacy_dict() -> None:
    workflow = YoloWorkflow(_StubYoloProcessor())

    result = workflow.get_dataset_statistics("/tmp/yolo")

    assert result["success"] is True
    assert result["statistics"]["is_valid"] is True


def test_yolo_workflow_clean_unmatched_files_returns_legacy_dict() -> None:
    workflow = YoloWorkflow(_StubYoloProcessor())

    result = workflow.clean_unmatched_files("/tmp/yolo", dry_run=True)

    assert result["success"] is True
    assert result["dry_run"] is True
    assert result["statistics"]["total_deleted"] == 0


def test_yolo_workflow_validate_classes_consistency_returns_legacy_dict() -> None:
    workflow = YoloWorkflow(_StubYoloProcessor())

    result = workflow.validate_classes_consistency(["/tmp/a", "/tmp/b"])

    assert result["success"] is True
    assert result["consistent"] is True
    assert result["classes"] == ["cat", "dog"]


def test_yolo_workflow_generate_output_name_returns_passthrough_string() -> None:
    workflow = YoloWorkflow(_StubYoloProcessor())

    result = workflow.generate_output_name(["cat", "dog"], ["/tmp/a", "/tmp/b"])

    assert result == "merged-cat-dog-2"


def test_yolo_workflow_merge_datasets_returns_legacy_dict() -> None:
    workflow = YoloWorkflow(_StubYoloProcessor())

    result = workflow.merge_datasets(
        dataset_paths=["/tmp/a", "/tmp/b"],
        output_path="/tmp/out",
        output_name="merged",
        image_prefix="img",
    )

    assert result["success"] is True
    assert result["output_path"] == "/tmp/out"
    assert result["output_name"] == "merged"
    assert result["merged_datasets"] == 2


def test_yolo_workflow_collect_all_classes_info_returns_passthrough_value() -> None:
    workflow = YoloWorkflow(_StubYoloProcessor())

    result = workflow.collect_all_classes_info(["/tmp/a", "/tmp/b"])

    assert isinstance(result, list)
    assert result[0]["classes"] == ["cat"]


def test_yolo_workflow_create_unified_class_mapping_returns_passthrough_value() -> None:
    workflow = YoloWorkflow(_StubYoloProcessor())

    result = workflow.create_unified_class_mapping(
        [
            {"dataset_path": "/tmp/a", "classes": ["cat"]},
            {"dataset_path": "/tmp/b", "classes": ["dog"]},
        ]
    )

    assert result[0] == ["cat", "dog"]
    assert result[1] == [{0: 0}, {0: 1}]


def test_yolo_workflow_generate_different_output_name_returns_passthrough_string() -> (
    None
):
    workflow = YoloWorkflow(_StubYoloProcessor())

    result = workflow.generate_different_output_name(
        ["cat", "dog"],
        ["/tmp/a", "/tmp/b"],
    )

    assert result == "different-cat-dog-2"


def test_yolo_workflow_merge_different_type_datasets_returns_legacy_dict() -> None:
    workflow = YoloWorkflow(_StubYoloProcessor())

    result = workflow.merge_different_type_datasets(
        dataset_paths=["/tmp/a", "/tmp/b"],
        output_path="/tmp/out",
        output_name="merged-diff",
        image_prefix="img",
        dataset_order=[1, 0],
    )

    assert result["success"] is True
    assert result["output_path"] == "/tmp/out"
    assert result["output_name"] == "merged-diff"
    assert result["dataset_order"] == [1, 0]


def test_yolo_workflow_process_ctds_dataset_returns_legacy_failure_on_exception() -> (
    None
):
    workflow = _StubYoloWorkflow(_StubYoloProcessorRaises())

    result = workflow.process_ctds_dataset("/tmp/input")

    assert isinstance(result, dict)
    assert result["success"] is False
    assert result["error_code"] == "INTERNAL_ERROR"
    assert result["error"] == "内部处理失败"
    assert result["error_details"]["exception_type"] == "RuntimeError"
