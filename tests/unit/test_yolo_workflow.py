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




def test_yolo_workflow_convert_yolo_to_ctds_dataset_returns_legacy_failure_on_exception() -> None:
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


def test_yolo_workflow_process_ctds_dataset_returns_legacy_failure_on_exception() -> None:
    workflow = _StubYoloWorkflow(_StubYoloProcessorRaises())

    result = workflow.process_ctds_dataset("/tmp/input")

    assert isinstance(result, dict)
    assert result["success"] is False
    assert result["error_code"] == "INTERNAL_ERROR"
    assert result["error"] == "内部处理失败"
    assert result["error_details"]["exception_type"] == "RuntimeError"
