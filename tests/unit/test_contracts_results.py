from integrated_script.contracts.results import OperationResult


def test_operation_result_from_legacy_maps_fields() -> None:
    legacy = {
        "success": True,
        "message": "ok",
        "output_path": "/tmp/out",
        "statistics": {"total": 1},
    }

    result = OperationResult.from_legacy(legacy)

    assert result.success is True
    assert result.message == "ok"
    assert result.payload["output_path"] == "/tmp/out"
    assert result.payload["statistics"]["total"] == 1


def test_operation_result_to_legacy_preserves_schema() -> None:
    legacy = {
        "success": False,
        "error": "failed",
        "error_code": "DATASET_ERROR",
        "stage": "pre_detection",
    }

    result = OperationResult.from_legacy(legacy)

    assert result.to_legacy() == legacy


def test_operation_result_failure_factory_returns_legacy_failure_shape() -> None:
    result = OperationResult.failure(message="bad", error_code="E_BAD")

    legacy = result.to_legacy()

    assert legacy["success"] is False
    assert legacy["error"] == "bad"
    assert legacy["error_code"] == "E_BAD"


def test_operation_result_to_legacy_prevents_payload_reserved_key_override() -> None:
    result = OperationResult.failure(
        message="bad",
        error_code="E_BAD",
        payload={"success": True, "error": "payload-error", "x": 1},
    )

    legacy = result.to_legacy()

    assert legacy["success"] is False
    assert legacy["error"] == "bad"
    assert legacy["error_code"] == "E_BAD"
    assert legacy["x"] == 1


def test_operation_result_from_legacy_uses_message_when_error_is_none() -> None:
    legacy = {
        "success": False,
        "error": None,
        "message": "fallback-message",
        "error_code": "E_FALLBACK",
    }

    result = OperationResult.from_legacy(legacy)

    assert result.message == "fallback-message"
