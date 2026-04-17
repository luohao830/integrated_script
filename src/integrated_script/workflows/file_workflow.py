from typing import Any, Dict, List, Optional

from ..contracts.errors import NormalizedError, normalize_exception
from ..contracts.results import OperationResult


class FileWorkflow:
    """File 相关 workflow 适配层。"""

    def __init__(self, processor: Any):
        self.processor = processor

    def organize_by_extension(
        self,
        source_dir: str,
        output_dir: Optional[str] = None,
        copy_files: bool = False,
    ) -> Dict[str, Any]:
        try:
            legacy_result = self.processor.organize_by_extension(
                source_dir,
                output_dir=output_dir,
                copy_files=copy_files,
            )
            return OperationResult.from_legacy(legacy_result).to_legacy()
        except Exception as error:  # noqa: BLE001
            normalized = self._normalize_error(error)
            return self._legacy_failure(normalized)

    def copy_files(
        self,
        source_dir: str,
        target_dir: str,
        file_patterns: Optional[List[str]] = None,
        recursive: bool = False,
        overwrite: bool = False,
        preserve_structure: bool = True,
    ) -> Dict[str, Any]:
        try:
            legacy_result = self.processor.copy_files(
                source_dir,
                target_dir,
                file_patterns=file_patterns,
                recursive=recursive,
                overwrite=overwrite,
                preserve_structure=preserve_structure,
            )
            return OperationResult.from_legacy(legacy_result).to_legacy()
        except Exception as error:  # noqa: BLE001
            normalized = self._normalize_error(error)
            return self._legacy_failure(normalized)

    def move_files(
        self,
        source_dir: str,
        target_dir: str,
        file_patterns: Optional[List[str]] = None,
        recursive: bool = False,
        overwrite: bool = False,
        preserve_structure: bool = True,
    ) -> Dict[str, Any]:
        try:
            legacy_result = self.processor.move_files(
                source_dir,
                target_dir,
                file_patterns=file_patterns,
                recursive=recursive,
                overwrite=overwrite,
                preserve_structure=preserve_structure,
            )
            return OperationResult.from_legacy(legacy_result).to_legacy()
        except Exception as error:  # noqa: BLE001
            normalized = self._normalize_error(error)
            return self._legacy_failure(normalized)

    def move_images_by_count(
        self,
        source_dir: str,
        target_dir: str,
        count: int,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        try:
            legacy_result = self.processor.move_images_by_count(
                source_dir,
                target_dir,
                count=count,
                overwrite=overwrite,
            )
            return OperationResult.from_legacy(legacy_result).to_legacy()
        except Exception as error:  # noqa: BLE001
            normalized = self._normalize_error(error)
            return self._legacy_failure(normalized)

    def delete_json_files_recursive(
        self,
        target_dir: str,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        try:
            legacy_result = self.processor.delete_json_files_recursive(
                target_dir,
                dry_run=dry_run,
            )
            return OperationResult.from_legacy(legacy_result).to_legacy()
        except Exception as error:  # noqa: BLE001
            normalized = self._normalize_error(error)
            return self._legacy_failure(normalized)

    def rename_images_labels_sync(
        self,
        images_dir: str,
        labels_dir: str,
        prefix: str,
        digits: int = 5,
        shuffle_order: bool = False,
    ) -> Dict[str, Any]:
        try:
            legacy_result = self.processor.rename_images_labels_sync(
                images_dir,
                labels_dir,
                prefix,
                digits,
                shuffle_order,
            )
            return OperationResult.from_legacy(legacy_result).to_legacy()
        except Exception as error:  # noqa: BLE001
            normalized = self._normalize_error(error)
            return self._legacy_failure(normalized)

    def rename_files_with_temp(
        self,
        source_dir: str,
        pattern: str,
        shuffle_order: bool = False,
    ) -> Dict[str, Any]:
        try:
            legacy_result = self.processor.rename_files_with_temp(
                source_dir,
                pattern,
                shuffle_order=shuffle_order,
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
