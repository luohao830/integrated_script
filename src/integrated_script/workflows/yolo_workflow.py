from typing import Any, Dict, Optional

from ..contracts.errors import NormalizedError, normalize_exception
from ..contracts.results import OperationResult


class YoloWorkflow:
    """YOLO 相关 workflow 适配层。

    Phase 1 目标：
    - 对内可使用统一 contracts
    - 对外继续返回 legacy dict（不破坏现有 UI）
    """

    def __init__(self, processor: Any):
        self.processor = processor

    def process_ctds_dataset(
        self,
        input_path: str,
        output_name: Optional[str] = None,
        keep_empty_labels: bool = False,
    ) -> Dict[str, Any]:
        try:
            legacy_result = self.processor.process_ctds_dataset(
                input_path,
                output_name=output_name,
                keep_empty_labels=keep_empty_labels,
            )
            return OperationResult.from_legacy(legacy_result).to_legacy()
        except Exception as error:  # noqa: BLE001
            normalized = self._normalize_error(error)
            return self._legacy_failure(normalized)

    def continue_ctds_processing(
        self,
        pre_result: Dict[str, Any],
        confirmed_type: str,
        keep_empty_labels: Optional[bool] = None,
    ) -> Dict[str, Any]:
        try:
            legacy_result = self.processor.continue_ctds_processing(
                pre_result,
                confirmed_type,
                keep_empty_labels=keep_empty_labels,
            )
            return OperationResult.from_legacy(legacy_result).to_legacy()
        except Exception as error:  # noqa: BLE001
            normalized = self._normalize_error(error)
            return self._legacy_failure(normalized)

    def convert_yolo_to_ctds_dataset(
        self,
        dataset_path: str,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            legacy_result = self.processor.convert_yolo_to_ctds_dataset(
                dataset_path,
                output_path=output_path,
            )
            return OperationResult.from_legacy(legacy_result).to_legacy()
        except Exception as error:  # noqa: BLE001
            normalized = self._normalize_error(error)
            return self._legacy_failure(normalized)

    def _normalize_error(self, error: Exception) -> NormalizedError:
        return normalize_exception(error)

    def _legacy_failure(self, normalized: NormalizedError) -> Dict[str, Any]:
        return OperationResult.failure(
            message=normalized.message,
            error_code=normalized.code,
            payload={"error_details": normalized.details},
        ).to_legacy()

