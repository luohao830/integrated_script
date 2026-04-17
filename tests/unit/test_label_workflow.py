from typing import Any, Dict

from integrated_script.contracts.errors import NormalizedError
from integrated_script.workflows.label_workflow import LabelWorkflow


class _StubLabelProcessor:
    def __init__(self) -> None:
        self.calls: Dict[str, Any] = {}

    def create_empty_labels(
        self,
        images_dir: str,
        labels_dir: str | None = None,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        self.calls["create_empty_labels"] = (images_dir, labels_dir, overwrite)
        return {"success": True, "images_dir": images_dir, "labels_dir": labels_dir}

    def flip_labels(
        self,
        labels_dir: str,
        flip_type: str = "horizontal",
        backup: bool = True,
    ) -> Dict[str, Any]:
        self.calls["flip_labels"] = (labels_dir, flip_type, backup)
        return {"success": True, "labels_dir": labels_dir, "flip_type": flip_type}

    def filter_labels_by_class(
        self,
        labels_dir: str,
        target_classes: list[int],
        action: str = "keep",
        backup: bool = True,
    ) -> Dict[str, Any]:
        self.calls["filter_labels_by_class"] = (
            labels_dir,
            target_classes,
            action,
            backup,
        )
        return {"success": True, "labels_dir": labels_dir, "target_classes": target_classes}

    def remove_empty_labels_and_images(
        self,
        dataset_dir: str,
        images_subdir: str = "images",
        labels_subdir: str = "labels",
    ) -> Dict[str, Any]:
        self.calls["remove_empty_labels_and_images"] = (
            dataset_dir,
            images_subdir,
            labels_subdir,
        )
        return {"success": True, "dataset_dir": dataset_dir}

    def remove_labels_with_only_class(
        self,
        dataset_dir: str,
        target_class: int,
        images_subdir: str = "images",
        labels_subdir: str = "labels",
    ) -> Dict[str, Any]:
        self.calls["remove_labels_with_only_class"] = (
            dataset_dir,
            target_class,
            images_subdir,
            labels_subdir,
        )
        return {
            "success": True,
            "dataset_dir": dataset_dir,
            "target_class": target_class,
        }


class _StubLabelWorkflow(LabelWorkflow):
    def _normalize_error(self, _error: Exception) -> NormalizedError:
        return NormalizedError(
            code="INTERNAL_ERROR",
            message="内部处理失败",
            details={"exception_type": "RuntimeError"},
        )


def test_label_workflow_create_empty_labels_returns_legacy_dict() -> None:
    processor = _StubLabelProcessor()
    workflow = LabelWorkflow(processor)

    result = workflow.create_empty_labels(
        "/tmp/images", labels_dir="/tmp/labels", overwrite=True
    )

    assert result["success"] is True
    assert processor.calls["create_empty_labels"] == ("/tmp/images", "/tmp/labels", True)


def test_label_workflow_flip_labels_returns_legacy_dict() -> None:
    processor = _StubLabelProcessor()
    workflow = LabelWorkflow(processor)

    result = workflow.flip_labels("/tmp/labels", flip_type="vertical", backup=False)

    assert result["success"] is True
    assert processor.calls["flip_labels"] == ("/tmp/labels", "vertical", False)


def test_label_workflow_filter_labels_by_class_returns_legacy_dict() -> None:
    processor = _StubLabelProcessor()
    workflow = LabelWorkflow(processor)

    result = workflow.filter_labels_by_class(
        "/tmp/labels", target_classes=[0, 1], action="remove", backup=False
    )

    assert result["success"] is True
    assert processor.calls["filter_labels_by_class"] == (
        "/tmp/labels",
        [0, 1],
        "remove",
        False,
    )


def test_label_workflow_remove_empty_labels_and_images_returns_legacy_dict() -> None:
    processor = _StubLabelProcessor()
    workflow = LabelWorkflow(processor)

    result = workflow.remove_empty_labels_and_images(
        "/tmp/dataset", images_subdir="imgs", labels_subdir="labs"
    )

    assert result["success"] is True
    assert processor.calls["remove_empty_labels_and_images"] == (
        "/tmp/dataset",
        "imgs",
        "labs",
    )


def test_label_workflow_remove_labels_with_only_class_returns_legacy_dict() -> None:
    processor = _StubLabelProcessor()
    workflow = LabelWorkflow(processor)

    result = workflow.remove_labels_with_only_class(
        "/tmp/dataset",
        target_class=2,
        images_subdir="imgs",
        labels_subdir="labs",
    )

    assert result["success"] is True
    assert processor.calls["remove_labels_with_only_class"] == (
        "/tmp/dataset",
        2,
        "imgs",
        "labs",
    )


def test_label_workflow_create_empty_labels_returns_legacy_failure_on_exception() -> None:
    class _StubLabelProcessorRaises(_StubLabelProcessor):
        def create_empty_labels(
            self,
            images_dir: str,
            labels_dir: str | None = None,
            overwrite: bool = False,
        ) -> Dict[str, Any]:
            del labels_dir, overwrite
            raise RuntimeError(f"create empty failed: {images_dir}")

    workflow = _StubLabelWorkflow(_StubLabelProcessorRaises())

    result = workflow.create_empty_labels("/tmp/images")

    assert result["success"] is False
    assert result["error_code"] == "INTERNAL_ERROR"
    assert result["error"] == "内部处理失败"
    assert result["error_details"]["exception_type"] == "RuntimeError"
