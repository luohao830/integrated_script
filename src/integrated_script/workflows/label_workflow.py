from typing import Any, Dict, List, Optional

from ..contracts.errors import NormalizedError, normalize_exception
from ..contracts.results import OperationResult


class LabelWorkflow:
    """Label 相关 workflow 适配层。"""

    def __init__(self, processor: Any):
        self.processor = processor

    def create_empty_labels(
        self,
        images_dir: str,
        labels_dir: Optional[str] = None,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        try:
            legacy_result = self.processor.create_empty_labels(
                images_dir,
                labels_dir=labels_dir,
                overwrite=overwrite,
            )
            return OperationResult.from_legacy(legacy_result).to_legacy()
        except Exception as error:  # noqa: BLE001
            normalized = self._normalize_error(error)
            return self._legacy_failure(normalized)

    def flip_labels(
        self,
        labels_dir: str,
        flip_type: str = "horizontal",
        backup: bool = True,
    ) -> Dict[str, Any]:
        try:
            legacy_result = self.processor.flip_labels(
                labels_dir,
                flip_type=flip_type,
                backup=backup,
            )
            return OperationResult.from_legacy(legacy_result).to_legacy()
        except Exception as error:  # noqa: BLE001
            normalized = self._normalize_error(error)
            return self._legacy_failure(normalized)

    def filter_labels_by_class(
        self,
        labels_dir: str,
        target_classes: List[int],
        action: str = "keep",
        backup: bool = True,
    ) -> Dict[str, Any]:
        try:
            legacy_result = self.processor.filter_labels_by_class(
                labels_dir,
                target_classes=target_classes,
                action=action,
                backup=backup,
            )
            return OperationResult.from_legacy(legacy_result).to_legacy()
        except Exception as error:  # noqa: BLE001
            normalized = self._normalize_error(error)
            return self._legacy_failure(normalized)

    def remove_empty_labels_and_images(
        self,
        dataset_dir: str,
        images_subdir: str = "images",
        labels_subdir: str = "labels",
    ) -> Dict[str, Any]:
        try:
            legacy_result = self.processor.remove_empty_labels_and_images(
                dataset_dir,
                images_subdir=images_subdir,
                labels_subdir=labels_subdir,
            )
            return OperationResult.from_legacy(legacy_result).to_legacy()
        except Exception as error:  # noqa: BLE001
            normalized = self._normalize_error(error)
            return self._legacy_failure(normalized)

    def remove_labels_with_only_class(
        self,
        dataset_dir: str,
        target_class: int,
        images_subdir: str = "images",
        labels_subdir: str = "labels",
    ) -> Dict[str, Any]:
        try:
            legacy_result = self.processor.remove_labels_with_only_class(
                dataset_dir,
                target_class=target_class,
                images_subdir=images_subdir,
                labels_subdir=labels_subdir,
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
