from typing import Any, Dict, Optional, Tuple

from ..contracts.errors import NormalizedError, normalize_exception
from ..contracts.results import OperationResult


class ImageWorkflow:
    """Image 相关 workflow 适配层。"""

    def __init__(self, processor: Any):
        self.processor = processor

    def convert_format(
        self,
        input_path: str,
        target_format: str,
        output_path: Optional[str] = None,
        quality: Optional[int] = None,
        recursive: bool = False,
    ) -> Dict[str, Any]:
        try:
            legacy_result = self.processor.convert_format(
                input_path,
                target_format,
                output_path=output_path,
                quality=quality,
                recursive=recursive,
            )
            return OperationResult.from_legacy(legacy_result).to_legacy()
        except Exception as error:  # noqa: BLE001
            normalized = self._normalize_error(error)
            return self._legacy_failure(normalized)

    def resize_images(
        self,
        input_dir: str,
        output_dir: str,
        target_size: Tuple[int, int],
        maintain_aspect_ratio: bool = True,
        recursive: bool = False,
    ) -> Dict[str, Any]:
        try:
            legacy_result = self.processor.resize_images(
                input_dir,
                output_dir,
                target_size=target_size,
                maintain_aspect_ratio=maintain_aspect_ratio,
                recursive=recursive,
            )
            return OperationResult.from_legacy(legacy_result).to_legacy()
        except Exception as error:  # noqa: BLE001
            normalized = self._normalize_error(error)
            return self._legacy_failure(normalized)

    def get_image_info(
        self,
        image_path: str,
        recursive: bool = False,
    ) -> Dict[str, Any]:
        try:
            legacy_result = self.processor.get_image_info(
                image_path,
                recursive=recursive,
            )
            return OperationResult.from_legacy(legacy_result).to_legacy()
        except Exception as error:  # noqa: BLE001
            normalized = self._normalize_error(error)
            return self._legacy_failure(normalized)

    def repair_images_with_opencv(
        self,
        directory: str,
        extensions=None,
        recursive: bool = False,
        include_hidden: bool = False,
    ) -> Dict[str, Any]:
        try:
            legacy_result = self.processor.repair_images_with_opencv(
                directory,
                extensions=extensions,
                recursive=recursive,
                include_hidden=include_hidden,
            )
            return OperationResult.from_legacy(legacy_result).to_legacy()
        except Exception as error:  # noqa: BLE001
            normalized = self._normalize_error(error)
            return self._legacy_failure(normalized)

    def compress_images_multiprocess_batch(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        quality: int = 85,
        target_format: Optional[str] = None,
        recursive: bool = False,
        max_size: Optional[Tuple[int, int]] = None,
        batch_count: int = 1,
        max_processes: Optional[int] = None,
    ) -> Dict[str, Any]:
        try:
            legacy_result = self.processor.compress_images_multiprocess_batch(
                input_dir=input_dir,
                output_dir=output_dir,
                quality=quality,
                target_format=target_format,
                recursive=recursive,
                max_size=max_size,
                batch_count=batch_count,
                max_processes=max_processes,
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
