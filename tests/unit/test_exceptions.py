import pytest

from integrated_script.config.exceptions import (
    ConfigurationError,
    DatasetError,
    FileProcessingError,
    PathError,
    ProcessingError,
    UserInterruptError,
    ValidationError,
)


def test_processing_error_formats_message_with_context() -> None:
    error = ProcessingError("failed", error_code="E", context={"k": "v"})

    assert str(error) == "failed (Code: E, Context: {'k': 'v'})"


def test_processing_error_uses_default_code_without_context() -> None:
    error = ProcessingError("failed")

    assert error.error_code == "PROCESSING_ERROR"
    assert str(error) == "failed (Code: PROCESSING_ERROR)"


@pytest.mark.parametrize(
    ("error", "field", "expected"),
    [
        (PathError("bad path", path="/tmp/a"), "path", "/tmp/a"),
        (
            FileProcessingError(
                "bad file",
                file_path="/tmp/a.txt",
                operation="copy",
            ),
            "operation",
            "copy",
        ),
        (
            ConfigurationError(
                "bad config",
                config_key="a.b",
                config_file="cfg.json",
            ),
            "config_key",
            "a.b",
        ),
        (
            ValidationError(
                "bad value",
                validation_type="range",
                expected="1",
                actual="2",
            ),
            "validation_type",
            "range",
        ),
        (
            DatasetError(
                "bad dataset",
                dataset_path="/tmp/ds",
                dataset_type="yolo",
            ),
            "dataset_type",
            "yolo",
        ),
    ],
)
def test_specialized_errors_store_typed_fields_and_context(
    error,
    field: str,
    expected: str,
) -> None:
    assert getattr(error, field) == expected


@pytest.mark.parametrize(
    "error,expected_context",
    [
        (PathError("bad path", path="/tmp/a"), {"path": "/tmp/a"}),
        (
            FileProcessingError(
                "bad file",
                file_path="/tmp/a.txt",
                operation="copy",
            ),
            {"file_path": "/tmp/a.txt", "operation": "copy"},
        ),
        (
            ConfigurationError(
                "bad config",
                config_key="a.b",
                config_file="cfg.json",
            ),
            {"config_key": "a.b", "config_file": "cfg.json"},
        ),
        (
            ValidationError(
                "bad value",
                validation_type="range",
                expected="1",
                actual="2",
            ),
            {
                "validation_type": "range",
                "expected": "1",
                "actual": "2",
            },
        ),
        (
            DatasetError(
                "bad dataset",
                dataset_path="/tmp/ds",
                dataset_type="yolo",
            ),
            {"dataset_path": "/tmp/ds", "dataset_type": "yolo"},
        ),
    ],
)
def test_specialized_errors_populate_context(
    error,
    expected_context: dict,
) -> None:
    assert error.context == expected_context


def test_user_interrupt_error_defaults_message_and_code() -> None:
    error = UserInterruptError()

    assert error.message == "操作被用户中断"
    assert error.error_code == "USER_INTERRUPT"
    assert str(error) == "操作被用户中断 (Code: USER_INTERRUPT)"
