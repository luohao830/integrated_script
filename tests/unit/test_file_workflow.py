from typing import Any, Dict, Optional

from integrated_script.contracts.errors import NormalizedError
from integrated_script.workflows.file_workflow import FileWorkflow


class _StubFileProcessor:
    def __init__(self) -> None:
        self.calls: Dict[str, Any] = {}

    def organize_by_extension(
        self,
        source_dir: str,
        output_dir: Optional[str] = None,
        copy_files: bool = False,
    ) -> Dict[str, Any]:
        self.calls["organize_by_extension"] = (source_dir, output_dir, copy_files)
        return {"success": True, "source_dir": source_dir, "target_dir": output_dir}

    def copy_files(
        self,
        source_dir: str,
        target_dir: str,
        file_patterns=None,
        recursive: bool = False,
        overwrite: bool = False,
        preserve_structure: bool = True,
    ) -> Dict[str, Any]:
        self.calls["copy_files"] = (
            source_dir,
            target_dir,
            file_patterns,
            recursive,
            overwrite,
            preserve_structure,
        )
        return {"success": True, "source_dir": source_dir, "target_dir": target_dir}

    def move_files(
        self,
        source_dir: str,
        target_dir: str,
        file_patterns=None,
        recursive: bool = False,
        overwrite: bool = False,
        preserve_structure: bool = True,
    ) -> Dict[str, Any]:
        self.calls["move_files"] = (
            source_dir,
            target_dir,
            file_patterns,
            recursive,
            overwrite,
            preserve_structure,
        )
        return {"success": True, "source_dir": source_dir, "target_dir": target_dir}

    def move_images_by_count(
        self,
        source_dir: str,
        target_dir: str,
        count: int,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        self.calls["move_images_by_count"] = (source_dir, target_dir, count, overwrite)
        return {"success": True, "source_dir": source_dir, "target_dir": target_dir}

    def delete_json_files_recursive(
        self,
        target_dir: str,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        self.calls["delete_json_files_recursive"] = (target_dir, dry_run)
        return {
            "success": True,
            "target_dir": target_dir,
            "dry_run": dry_run,
            "json_files": [],
            "statistics": {"total_files": 0, "deleted_count": 0, "failed_count": 0},
        }

    def rename_images_labels_sync(
        self,
        images_dir: str,
        labels_dir: str,
        prefix: str,
        digits: int = 5,
        shuffle_order: bool = False,
    ) -> Dict[str, Any]:
        self.calls["rename_images_labels_sync"] = (
            images_dir,
            labels_dir,
            prefix,
            digits,
            shuffle_order,
        )
        return {"success": True, "renamed_count": 1}

    def rename_files_with_temp(
        self,
        source_dir: str,
        pattern: str,
        shuffle_order: bool = False,
    ) -> Dict[str, Any]:
        self.calls["rename_files_with_temp"] = (source_dir, pattern, shuffle_order)
        return {"success": True, "renamed_count": 1}


class _StubFileWorkflow(FileWorkflow):
    def _normalize_error(self, _error: Exception) -> NormalizedError:
        return NormalizedError(
            code="INTERNAL_ERROR",
            message="内部处理失败",
            details={"exception_type": "RuntimeError"},
        )


def test_file_workflow_organize_by_extension_returns_legacy_dict() -> None:
    processor = _StubFileProcessor()
    workflow = FileWorkflow(processor)

    result = workflow.organize_by_extension(
        "/tmp/src", output_dir="/tmp/out", copy_files=True
    )

    assert result["success"] is True
    assert processor.calls["organize_by_extension"] == ("/tmp/src", "/tmp/out", True)


def test_file_workflow_copy_files_returns_legacy_dict() -> None:
    processor = _StubFileProcessor()
    workflow = FileWorkflow(processor)

    result = workflow.copy_files(
        "/tmp/src",
        "/tmp/out",
        file_patterns=["*.jpg"],
        recursive=True,
        overwrite=True,
        preserve_structure=False,
    )

    assert result["success"] is True
    assert processor.calls["copy_files"] == (
        "/tmp/src",
        "/tmp/out",
        ["*.jpg"],
        True,
        True,
        False,
    )


def test_file_workflow_move_files_returns_legacy_dict() -> None:
    processor = _StubFileProcessor()
    workflow = FileWorkflow(processor)

    result = workflow.move_files(
        "/tmp/src",
        "/tmp/out",
        file_patterns=["*.txt"],
        recursive=False,
        overwrite=False,
        preserve_structure=True,
    )

    assert result["success"] is True
    assert processor.calls["move_files"] == (
        "/tmp/src",
        "/tmp/out",
        ["*.txt"],
        False,
        False,
        True,
    )


def test_file_workflow_move_images_by_count_returns_legacy_dict() -> None:
    processor = _StubFileProcessor()
    workflow = FileWorkflow(processor)

    result = workflow.move_images_by_count(
        "/tmp/src", "/tmp/out", count=10, overwrite=True
    )

    assert result["success"] is True
    assert processor.calls["move_images_by_count"] == ("/tmp/src", "/tmp/out", 10, True)


def test_file_workflow_delete_json_files_recursive_returns_legacy_dict() -> None:
    processor = _StubFileProcessor()
    workflow = FileWorkflow(processor)

    result = workflow.delete_json_files_recursive("/tmp/dir", dry_run=True)

    assert result["success"] is True
    assert result["dry_run"] is True
    assert processor.calls["delete_json_files_recursive"] == ("/tmp/dir", True)


def test_file_workflow_rename_images_labels_sync_returns_legacy_dict() -> None:
    processor = _StubFileProcessor()
    workflow = FileWorkflow(processor)

    result = workflow.rename_images_labels_sync(
        "/tmp/images", "/tmp/labels", "demo", digits=6, shuffle_order=True
    )

    assert result["success"] is True
    assert processor.calls["rename_images_labels_sync"] == (
        "/tmp/images",
        "/tmp/labels",
        "demo",
        6,
        True,
    )


def test_file_workflow_rename_files_with_temp_returns_legacy_dict() -> None:
    processor = _StubFileProcessor()
    workflow = FileWorkflow(processor)

    result = workflow.rename_files_with_temp(
        "/tmp/src", "{index}.jpg", shuffle_order=True
    )

    assert result["success"] is True
    assert processor.calls["rename_files_with_temp"] == (
        "/tmp/src",
        "{index}.jpg",
        True,
    )


def test_file_workflow_copy_files_returns_legacy_failure_on_exception() -> None:
    class _StubFileProcessorRaises(_StubFileProcessor):
        def copy_files(
            self,
            source_dir: str,
            target_dir: str,
            file_patterns=None,
            recursive: bool = False,
            overwrite: bool = False,
            preserve_structure: bool = True,
        ) -> Dict[str, Any]:
            del target_dir, file_patterns, recursive, overwrite, preserve_structure
            raise RuntimeError(f"copy failed: {source_dir}")

    workflow = _StubFileWorkflow(_StubFileProcessorRaises())

    result = workflow.copy_files("/tmp/src", "/tmp/out")

    assert result["success"] is False
    assert result["error_code"] == "INTERNAL_ERROR"
    assert result["error"] == "内部处理失败"
    assert result["error_details"]["exception_type"] == "RuntimeError"
