from integrated_script.config.exceptions import DatasetError
from integrated_script.contracts.errors import normalize_exception


def test_normalize_exception_uses_processing_error_code_and_context() -> None:
    error = DatasetError("bad dataset", dataset_path="/tmp/ds")

    normalized = normalize_exception(error)

    assert normalized.code == "DATASET_ERROR"
    assert normalized.message == "bad dataset"
    assert normalized.details["dataset_path"] == "/tmp/ds"


def test_normalize_exception_masks_unexpected_error_message() -> None:
    error = RuntimeError("secret-token-123")

    normalized = normalize_exception(error)

    assert normalized.code == "INTERNAL_ERROR"
    assert normalized.message == "内部处理失败"
    assert normalized.details["exception_type"] == "RuntimeError"
    assert "secret-token" not in normalized.message
