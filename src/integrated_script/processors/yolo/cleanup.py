"""Cleanup and statistics internals for YOLO processor Phase 3 migration slices."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, cast

from ...config.exceptions import DatasetError
from ...core.progress import progress_context
from ...core.utils import delete_file_safe, validate_path

if TYPE_CHECKING:
    from ..yolo_processor import YOLOProcessor


def get_dataset_statistics_internal(
    processor: "YOLOProcessor", dataset_path: str
) -> Dict[str, Any]:
    """获取数据集统计信息。"""
    try:
        original_dataset_path = dataset_path
        dataset_dir = validate_path(dataset_path, must_exist=True, must_be_dir=True)

        dataset_dir = processor._detect_dataset_root(dataset_dir)

        processor.logger.info(f"开始获取YOLO数据集统计信息: {dataset_dir}")

        validation_result = processor._validate_yolo_dataset(
            dataset_dir, check_integrity=True
        )

        classes_file = dataset_dir / processor.classes_file
        classes_file_path = str(classes_file) if classes_file.exists() else None

        stats = validation_result["statistics"].copy()
        stats["dataset_path"] = str(dataset_dir)
        stats["original_path"] = original_dataset_path
        stats["is_valid"] = validation_result.get("success", False)
        stats["has_classes_file"] = classes_file_path is not None

        if classes_file_path:
            try:
                with open(classes_file_path, "r", encoding="utf-8") as f:
                    classes = [line.strip() for line in f.readlines() if line.strip()]
                stats["num_classes"] = len(classes)
                stats["class_names"] = classes
            except Exception:
                stats["num_classes"] = 0
                stats["class_names"] = []
        else:
            stats["num_classes"] = 0
            stats["class_names"] = []

        result: Dict[str, Any] = {
            "statistics": stats,
            "valid": validation_result.get("success", False),
            "classes_file": classes_file_path,
            "issues": validation_result.get("issues", []),
        }

        return result

    except Exception as e:
        raise DatasetError(
            f"获取数据集统计信息失败: {str(e)}", dataset_path=dataset_path
        )


def clean_unmatched_files_internal(
    processor: "YOLOProcessor", dataset_path: str, dry_run: bool = False
) -> Dict[str, Any]:
    """清理数据集中不匹配的文件。"""
    try:
        dataset_dir = validate_path(dataset_path, must_exist=True, must_be_dir=True)
        processor.logger.info(f"开始清理不匹配文件: {dataset_dir}")

        validation_result: Dict[str, Any] = processor.get_dataset_statistics(
            dataset_path
        )

        deleted_files: Dict[str, List[str]] = {
            "orphaned_images": [],
            "orphaned_labels": [],
            "invalid_labels": [],
            "empty_labels": [],
        }
        statistics: Dict[str, int] = {
            "total_deleted": 0,
            "deleted_images": 0,
            "deleted_labels": 0,
        }
        result: Dict[str, Any] = {
            "success": True,
            "dataset_path": str(dataset_dir),
            "dry_run": dry_run,
            "deleted_files": deleted_files,
            "statistics": statistics,
        }

        if validation_result["valid"]:
            processor.logger.info("数据集已经有效，无需清理")
            return result

        files_to_delete: List[Tuple[Path, str]] = []
        issues: List[Dict[str, Any]] = cast(
            List[Dict[str, Any]], validation_result.get("issues", [])
        )

        processor.logger.info(f"验证结果issues数量: {len(issues)}")
        for i, issue in enumerate(issues):
            processor.logger.info(
                "Issue %s: type=%s, count=%s, files_count=%s",
                i,
                issue.get("type"),
                issue.get("count"),
                len(issue.get("files", [])),
            )

        for issue in issues:
            issue_type = issue.get("type")
            if issue_type == "orphaned_images":
                for img_path in cast(List[str], issue.get("files", [])):
                    files_to_delete.append((Path(img_path), "orphaned_images"))
            elif issue_type == "orphaned_labels":
                for lbl_path in cast(List[str], issue.get("files", [])):
                    files_to_delete.append((Path(lbl_path), "orphaned_labels"))
            elif issue_type == "invalid_labels":
                for invalid_label in cast(
                    List[Dict[str, Any]], issue.get("examples", [])
                ):
                    files_to_delete.append(
                        (Path(invalid_label["file"]), "invalid_labels")
                    )
            elif issue_type == "empty_labels":
                for empty_label_path in cast(List[str], issue.get("files", [])):
                    files_to_delete.append((Path(empty_label_path), "empty_labels"))

        with progress_context(len(files_to_delete), "清理不匹配文件") as progress:
            for file_path, file_type in files_to_delete:
                try:
                    deleted_bucket = deleted_files[file_type]
                    if dry_run:
                        deleted_bucket.append(str(file_path))
                        processor.logger.info(f"[试运行] 将删除: {file_path}")
                    else:
                        if file_path.exists():
                            delete_file_safe(file_path)
                            deleted_bucket.append(str(file_path))
                            processor.logger.info(f"已删除: {file_path}")

                            if file_type == "orphaned_images":
                                statistics["deleted_images"] += 1
                            else:
                                statistics["deleted_labels"] += 1

                    progress.update_progress(1)

                except Exception as e:
                    processor.logger.error(f"删除文件失败 {file_path}: {str(e)}")
                    result["success"] = False

        statistics["total_deleted"] = (
            statistics["deleted_images"] + statistics["deleted_labels"]
        )

        if dry_run:
            processor.logger.info(f"试运行完成，将删除 {len(files_to_delete)} 个文件")
        else:
            processor.logger.info(f"清理完成: {result['statistics']}")

        return result

    except Exception as e:
        raise DatasetError(f"清理不匹配文件失败: {str(e)}", dataset_path=dataset_path)
