from typing import Any, Dict, List, Optional, Tuple

import pytest

from integrated_script.config.settings import ConfigManager
from integrated_script.ui.interactive import InteractiveInterface


class _StubYoloProcessor:
    def __init__(
        self,
        pre_result: Dict[str, Any],
        continue_result: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.pre_result = pre_result
        self.continue_result = continue_result or {"success": True}
        self.process_called = False
        self.continue_called = False
        self.last_continue_args: Optional[Tuple[Dict[str, Any], str]] = None
        self.last_continue_kwargs: Dict[str, Any] = {}

    def process_ctds_dataset(self, _dataset_path: str, **_kwargs: Any) -> Dict[str, Any]:
        self.process_called = True
        return self.pre_result

    def continue_ctds_processing(
        self,
        _pre_result: Dict[str, Any],
        _confirmed_type: str,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        self.continue_called = True
        self.last_continue_args = (_pre_result, _confirmed_type)
        self.last_continue_kwargs = _kwargs
        return self.continue_result

    def convert_yolo_to_ctds_dataset(
        self,
        dataset_path: str,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
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


class _RecordingYoloWorkflow:
    instances: List["_RecordingYoloWorkflow"] = []

    def __init__(self, processor: Any) -> None:
        self.processor = processor
        self.process_calls: List[Tuple[str, Optional[str], bool]] = []
        self.continue_calls: List[Tuple[Dict[str, Any], str, Optional[bool]]] = []
        self.convert_calls: List[Tuple[str, Optional[str]]] = []
        _RecordingYoloWorkflow.instances.append(self)

    def process_ctds_dataset(
        self,
        input_path: str,
        output_name: Optional[str] = None,
        keep_empty_labels: bool = False,
    ) -> Dict[str, Any]:
        self.process_calls.append((input_path, output_name, keep_empty_labels))
        return {
            "success": True,
            "stage": "pre_detection",
            "pre_detection_result": {
                "dataset_type": "detection",
                "confidence": 0.95,
            },
        }

    def continue_ctds_processing(
        self,
        pre_result: Dict[str, Any],
        confirmed_type: str,
        keep_empty_labels: Optional[bool] = None,
    ) -> Dict[str, Any]:
        self.continue_calls.append((pre_result, confirmed_type, keep_empty_labels))
        return {
            "success": True,
            "output_path": "/tmp/output",
            "project_name": "demo",
            "statistics": {"final_count": 1},
        }

    def convert_yolo_to_ctds_dataset(
        self,
        dataset_path: str,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.convert_calls.append((dataset_path, output_path))
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


@pytest.fixture(autouse=True)
def _reset_recording_workflow_state() -> None:
    _RecordingYoloWorkflow.instances.clear()


@pytest.fixture
def interface_with_non_exe(monkeypatch, tmp_path) -> InteractiveInterface:
    monkeypatch.setattr(InteractiveInterface, "_is_running_as_exe", lambda self: False)
    config = ConfigManager(config_file=tmp_path / "config.json", auto_save=False)
    return InteractiveInterface(config_manager=config)


@pytest.fixture
def interface_with_exe(monkeypatch, tmp_path) -> InteractiveInterface:
    monkeypatch.setattr(InteractiveInterface, "_is_running_as_exe", lambda self: True)
    config = ConfigManager(config_file=tmp_path / "config.json", auto_save=False)
    return InteractiveInterface(config_manager=config)


def test_main_menu_includes_environment_entry_when_not_exe(interface_with_non_exe) -> None:
    options = [name for name, _ in interface_with_non_exe.menu_system.main_menu["options"]]

    assert "环境检查与配置" in options


def test_main_menu_hides_environment_entry_when_exe(interface_with_exe) -> None:
    options = [name for name, _ in interface_with_exe.menu_system.main_menu["options"]]

    assert "环境检查与配置" not in options


def test_get_processor_reuses_cached_instance(interface_with_non_exe) -> None:
    processor1 = interface_with_non_exe._get_processor("file")
    processor2 = interface_with_non_exe._get_processor("file")

    assert processor1 is processor2


def test_get_processor_unknown_key_raises_key_error(interface_with_non_exe) -> None:
    with pytest.raises(KeyError):
        interface_with_non_exe._get_processor("unknown")


def test_get_user_confirmed_type_returns_none_when_low_confidence_user_cancels(
    interface_with_non_exe, monkeypatch
) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt="": "3")

    confirmed = interface_with_non_exe._get_user_confirmed_type(
        detected_type="mixed", confidence=0.5
    )

    assert confirmed is None


def test_get_user_confirmed_type_returns_detected_type_when_high_confidence_confirmed(
    interface_with_non_exe, monkeypatch
) -> None:
    monkeypatch.setattr(interface_with_non_exe, "_get_yes_no_input", lambda *_a, **_k: True)

    confirmed = interface_with_non_exe._get_user_confirmed_type(
        detected_type="detection", confidence=0.95
    )

    assert confirmed == "detection"


def test_yolo_process_ctds_aborts_when_user_cancels_confirmation(
    interface_with_non_exe, monkeypatch
) -> None:
    pre_result = {
        "stage": "pre_detection",
        "pre_detection_result": {"dataset_type": "detection", "confidence": 0.95},
    }
    processor = _StubYoloProcessor(pre_result=pre_result)

    monkeypatch.setattr(
        interface_with_non_exe,
        "_get_path_input",
        lambda _prompt, must_exist=True: "/tmp/ctds-dataset",
    )
    monkeypatch.setattr(interface_with_non_exe, "_get_processor", lambda _name: processor)
    monkeypatch.setattr(interface_with_non_exe, "_get_yes_no_input", lambda *_a, **_k: False)
    monkeypatch.setattr(
        interface_with_non_exe,
        "_get_user_confirmed_type",
        lambda _detected_type, _confidence: None,
    )
    monkeypatch.setattr(interface_with_non_exe, "_pause", lambda: None)
    inputs = iter([""])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    interface_with_non_exe._yolo_process_ctds()

    assert processor.process_called is True
    assert processor.continue_called is False


def test_yolo_process_ctds_continues_after_user_confirmation(
    interface_with_non_exe, monkeypatch
) -> None:
    pre_result = {
        "stage": "pre_detection",
        "pre_detection_result": {"dataset_type": "detection", "confidence": 0.95},
    }
    continue_result = {
        "success": True,
        "output_path": "/tmp/output",
        "project_name": "demo",
        "statistics": {"final_count": 1},
    }
    processor = _StubYoloProcessor(pre_result=pre_result, continue_result=continue_result)
    displayed = {}

    monkeypatch.setattr(
        interface_with_non_exe,
        "_get_path_input",
        lambda _prompt, must_exist=True: "/tmp/ctds-dataset",
    )
    monkeypatch.setattr(interface_with_non_exe, "_get_processor", lambda _name: processor)
    monkeypatch.setattr(interface_with_non_exe, "_get_yes_no_input", lambda *_a, **_k: False)
    monkeypatch.setattr(
        interface_with_non_exe,
        "_get_user_confirmed_type",
        lambda _detected_type, _confidence: "detection",
    )
    monkeypatch.setattr(
        interface_with_non_exe,
        "_display_ctds_result",
        lambda result: displayed.setdefault("result", result),
    )
    monkeypatch.setattr(interface_with_non_exe, "_pause", lambda: None)
    inputs = iter([""])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    interface_with_non_exe._yolo_process_ctds()

    assert processor.process_called is True
    assert processor.continue_called is True
    assert processor.last_continue_args is not None
    assert processor.last_continue_args[1] == "detection"
    assert processor.last_continue_kwargs.get("keep_empty_labels") is False
    assert displayed["result"] == continue_result


def test_yolo_process_ctds_uses_yolo_workflow_adapter(
    interface_with_non_exe, monkeypatch
) -> None:
    marker_processor = object()
    displayed: Dict[str, Any] = {}

    monkeypatch.setattr(
        "integrated_script.ui.interactive.YoloWorkflow",
        _RecordingYoloWorkflow,
    )
    monkeypatch.setattr(
        interface_with_non_exe,
        "_get_path_input",
        lambda _prompt, must_exist=True: "/tmp/ctds-dataset",
    )
    monkeypatch.setattr(interface_with_non_exe, "_get_processor", lambda _name: marker_processor)
    monkeypatch.setattr(interface_with_non_exe, "_get_yes_no_input", lambda *_a, **_k: False)
    monkeypatch.setattr(
        interface_with_non_exe,
        "_get_user_confirmed_type",
        lambda _detected_type, _confidence: "detection",
    )
    monkeypatch.setattr(
        interface_with_non_exe,
        "_display_ctds_result",
        lambda result: displayed.setdefault("result", result),
    )
    monkeypatch.setattr(interface_with_non_exe, "_pause", lambda: None)
    inputs = iter([""])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    interface_with_non_exe._yolo_process_ctds()

    assert len(_RecordingYoloWorkflow.instances) == 1
    workflow = _RecordingYoloWorkflow.instances[0]
    assert workflow.processor is marker_processor
    assert len(workflow.process_calls) == 1
    assert len(workflow.continue_calls) == 1
    assert workflow.continue_calls[0][1] == "detection"
    assert workflow.continue_calls[0][2] is False
    assert displayed["result"]["success"] is True
    assert displayed["result"]["project_name"] == "demo"


def test_yolo_convert_to_ctds_uses_yolo_workflow_adapter(
    interface_with_non_exe, monkeypatch
) -> None:
    marker_processor = object()
    inputs = iter(["/tmp/ctds-out"])

    monkeypatch.setattr(
        "integrated_script.ui.interactive.YoloWorkflow",
        _RecordingYoloWorkflow,
    )
    monkeypatch.setattr(
        interface_with_non_exe,
        "_get_path_input",
        lambda _prompt, must_exist=True: "/tmp/yolo-dataset",
    )
    monkeypatch.setattr(interface_with_non_exe, "_get_processor", lambda _name: marker_processor)
    monkeypatch.setattr(interface_with_non_exe, "_pause", lambda: None)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    interface_with_non_exe._yolo_convert_to_ctds()

    assert len(_RecordingYoloWorkflow.instances) == 1
    workflow = _RecordingYoloWorkflow.instances[0]
    assert workflow.processor is marker_processor
    assert len(workflow.convert_calls) == 1
    assert workflow.convert_calls[0][0] == "/tmp/yolo-dataset"
    assert workflow.convert_calls[0][1] == "/tmp/ctds-out"


def test_yolo_convert_to_ctds_shows_failure_message_from_workflow(
    interface_with_non_exe, monkeypatch
) -> None:
    class _FailingWorkflow(_RecordingYoloWorkflow):
        def convert_yolo_to_ctds_dataset(
            self,
            dataset_path: str,
            output_path: Optional[str] = None,
        ) -> Dict[str, Any]:
            self.convert_calls.append((dataset_path, output_path))
            return {
                "success": False,
                "error": "内部处理失败",
                "error_code": "INTERNAL_ERROR",
            }

    marker_processor = object()
    inputs = iter([""])
    printed: List[str] = []

    monkeypatch.setattr(
        "integrated_script.ui.interactive.YoloWorkflow",
        _FailingWorkflow,
    )
    monkeypatch.setattr(
        interface_with_non_exe,
        "_get_path_input",
        lambda _prompt, must_exist=True: "/tmp/yolo-dataset",
    )
    monkeypatch.setattr(interface_with_non_exe, "_get_processor", lambda _name: marker_processor)
    monkeypatch.setattr(interface_with_non_exe, "_pause", lambda: None)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))
    monkeypatch.setattr(
        "builtins.print",
        lambda *args, **_kwargs: printed.append(" ".join(str(arg) for arg in args)),
    )

    interface_with_non_exe._yolo_convert_to_ctds()

    assert any("❌ 转换失败" in line for line in printed)
    assert any("错误信息: 内部处理失败" in line for line in printed)
    assert len(_FailingWorkflow.instances) == 1
    assert _FailingWorkflow.instances[0].convert_calls[0][0] == "/tmp/yolo-dataset"
    assert _FailingWorkflow.instances[0].convert_calls[0][1] is None
    assert _FailingWorkflow.instances[0].processor is marker_processor
