from typing import Any, Dict, Optional, Tuple

from integrated_script.contracts.errors import NormalizedError
from integrated_script.workflows.image_workflow import ImageWorkflow


class _StubImageProcessor:
    def __init__(self) -> None:
        self.calls: Dict[str, Tuple[Any, ...]] = {}

    def convert_format(
        self,
        input_path: str,
        target_format: str,
        output_path: Optional[str] = None,
        quality: Optional[int] = None,
        recursive: bool = False,
    ) -> Dict[str, Any]:
        self.calls["convert_format"] = (
            input_path,
            target_format,
            output_path,
            quality,
            recursive,
        )
        return {
            "success": True,
            "input_path": input_path,
            "target_format": target_format,
            "output_path": output_path,
        }

    def resize_images(
        self,
        input_dir: str,
        output_dir: str,
        target_size: Tuple[int, int],
        maintain_aspect_ratio: bool = True,
        recursive: bool = False,
    ) -> Dict[str, Any]:
        self.calls["resize_images"] = (
            input_dir,
            output_dir,
            target_size,
            maintain_aspect_ratio,
            recursive,
        )
        return {
            "success": True,
            "input_dir": input_dir,
            "output_dir": output_dir,
            "target_size": target_size,
        }

    def get_image_info(self, image_path: str, recursive: bool = False) -> Dict[str, Any]:
        self.calls["get_image_info"] = (image_path, recursive)
        return {"success": True, "file_path": image_path}

    def repair_images_with_opencv(
        self,
        directory: str,
        extensions=None,
        recursive: bool = False,
        include_hidden: bool = False,
    ) -> Dict[str, Any]:
        self.calls["repair_images_with_opencv"] = (
            directory,
            extensions,
            recursive,
            include_hidden,
        )
        return {"success": True, "directory": directory}

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
        self.calls["compress_images_multiprocess_batch"] = (
            input_dir,
            output_dir,
            quality,
            target_format,
            recursive,
            max_size,
            batch_count,
            max_processes,
        )
        return {"success": True, "input_dir": input_dir, "output_dir": output_dir}


class _StubImageWorkflow(ImageWorkflow):
    def _normalize_error(self, _error: Exception) -> NormalizedError:
        return NormalizedError(
            code="INTERNAL_ERROR",
            message="内部处理失败",
            details={"exception_type": "RuntimeError"},
        )


def test_image_workflow_convert_format_returns_legacy_dict() -> None:
    processor = _StubImageProcessor()
    workflow = ImageWorkflow(processor)

    result = workflow.convert_format(
        "/tmp/input.jpg",
        "png",
        output_path="/tmp/out",
        quality=90,
        recursive=True,
    )

    assert result["success"] is True
    assert result["target_format"] == "png"
    assert processor.calls["convert_format"] == (
        "/tmp/input.jpg",
        "png",
        "/tmp/out",
        90,
        True,
    )


def test_image_workflow_resize_images_returns_legacy_dict() -> None:
    processor = _StubImageProcessor()
    workflow = ImageWorkflow(processor)

    result = workflow.resize_images(
        "/tmp/in",
        "/tmp/out",
        target_size=(640, 480),
        maintain_aspect_ratio=False,
        recursive=True,
    )

    assert result["success"] is True
    assert result["target_size"] == (640, 480)
    assert processor.calls["resize_images"] == (
        "/tmp/in",
        "/tmp/out",
        (640, 480),
        False,
        True,
    )


def test_image_workflow_get_image_info_returns_legacy_dict() -> None:
    processor = _StubImageProcessor()
    workflow = ImageWorkflow(processor)

    result = workflow.get_image_info("/tmp/a.jpg", recursive=True)

    assert result["success"] is True
    assert result["file_path"] == "/tmp/a.jpg"
    assert processor.calls["get_image_info"] == ("/tmp/a.jpg", True)


def test_image_workflow_repair_images_with_opencv_returns_legacy_dict() -> None:
    processor = _StubImageProcessor()
    workflow = ImageWorkflow(processor)

    result = workflow.repair_images_with_opencv(
        "/tmp/images",
        extensions=[".jpg"],
        recursive=True,
        include_hidden=True,
    )

    assert result["success"] is True
    assert result["directory"] == "/tmp/images"
    assert processor.calls["repair_images_with_opencv"] == (
        "/tmp/images",
        [".jpg"],
        True,
        True,
    )


def test_image_workflow_compress_images_multiprocess_batch_returns_legacy_dict() -> None:
    processor = _StubImageProcessor()
    workflow = ImageWorkflow(processor)

    result = workflow.compress_images_multiprocess_batch(
        input_dir="/tmp/images",
        output_dir="/tmp/out",
        quality=80,
        target_format="jpg",
        recursive=True,
        max_size=(1024, 1024),
        batch_count=4,
        max_processes=2,
    )

    assert result["success"] is True
    assert result["input_dir"] == "/tmp/images"
    assert processor.calls["compress_images_multiprocess_batch"] == (
        "/tmp/images",
        "/tmp/out",
        80,
        "jpg",
        True,
        (1024, 1024),
        4,
        2,
    )


def test_image_workflow_convert_format_returns_legacy_failure_on_exception() -> None:
    class _StubImageProcessorRaises(_StubImageProcessor):
        def convert_format(
            self,
            input_path: str,
            target_format: str,
            output_path: Optional[str] = None,
            quality: Optional[int] = None,
            recursive: bool = False,
        ) -> Dict[str, Any]:
            del target_format, output_path, quality, recursive
            raise RuntimeError(f"convert failed: {input_path}")

    workflow = _StubImageWorkflow(_StubImageProcessorRaises())
    result = workflow.convert_format("/tmp/input.jpg", "png")

    assert result["success"] is False
    assert result["error_code"] == "INTERNAL_ERROR"
    assert result["error"] == "内部处理失败"
    assert result["error_details"]["exception_type"] == "RuntimeError"
