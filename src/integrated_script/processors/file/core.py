#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
file_processor.py

文件处理器

提供文件复制、移动、删除、重命名等基本文件操作功能。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from ...config.exceptions import ProcessingError
from ...core.base import BaseProcessor
from ...core.progress import process_with_progress
from ...core.utils import (
    copy_file_safe,
    create_directory,
    format_file_size,
    get_file_list,
    get_unique_filename,
    move_file_safe,
    validate_path,
)


class FileProcessor(BaseProcessor):
    """文件处理器

    提供文件复制、移动、删除、重命名等基本文件操作功能。

    """

    def __init__(self, **kwargs):
        """初始化文件处理器"""
        super().__init__(name="FileProcessor", **kwargs)

    def initialize(self) -> None:
        """初始化处理器"""
        self.logger.info("文件处理器初始化完成")

    def _build_safe_rename_target(self, base_dir: Path, file_name: str) -> Path:
        """构建安全的重命名目标路径（仅允许目录内文件名）。"""
        if not file_name:
            raise ValueError("文件名不能为空")

        if "/" in file_name or "\\" in file_name:
            raise ValueError(f"文件名不能包含路径分隔符: {file_name}")

        name_path = Path(file_name)
        if name_path.is_absolute() or name_path.name != file_name:
            raise ValueError(f"不安全的文件名: {file_name}")

        candidate = base_dir / file_name
        if candidate.resolve(strict=False).parent != base_dir.resolve():
            raise ValueError(f"重命名目标超出目录范围: {file_name}")

        return candidate

    def process(self, *args, **kwargs) -> Any:
        """主要处理方法（由子方法实现具体功能）"""
        raise NotImplementedError("请使用具体的处理方法")

    def copy_files(
        self,
        source_dir: str,
        target_dir: str,
        file_patterns: Optional[List[str]] = None,
        recursive: bool = False,
        overwrite: bool = False,
        preserve_structure: bool = True,
    ) -> Dict[str, Any]:
        """批量复制文件

        Args:
            source_dir: 源目录
            target_dir: 目标目录
            file_patterns: 文件模式列表（如 ['*.txt', '*.py']）
            recursive: 是否递归处理
            overwrite: 是否覆盖已存在的文件
            preserve_structure: 是否保持目录结构

        Returns:
            Dict[str, Any]: 复制结果
        """
        try:
            source_path = validate_path(source_dir, must_exist=True, must_be_dir=True)
            target_path = validate_path(target_dir, must_exist=False)

            create_directory(target_path)

            self.logger.info(f"开始复制文件: {source_path} -> {target_path}")
            self.logger.info(
                f"递归: {recursive}, 覆盖: {overwrite}, 保持结构: {preserve_structure}"
            )

            # 获取文件列表
            if file_patterns:
                files_to_copy: List[Path] = []
                for pattern in file_patterns:
                    if recursive:
                        files_to_copy.extend(source_path.rglob(pattern))
                    else:
                        files_to_copy.extend(source_path.glob(pattern))
                # 去重并过滤文件
                files_to_copy = [f for f in set(files_to_copy) if f.is_file()]
            else:
                # 获取所有文件
                files_to_copy = get_file_list(
                    source_path, extensions=None, recursive=recursive
                )

            result: Dict[str, Any] = {
                "success": True,
                "source_dir": str(source_path),
                "target_dir": str(target_path),
                "copied_files": [],
                "failed_files": [],
                "skipped_files": [],
                "statistics": {
                    "total_files": len(files_to_copy),
                    "copied_count": 0,
                    "failed_count": 0,
                    "skipped_count": 0,
                    "total_size": 0,
                },
            }

            # 复制文件
            def copy_single_file(file_path: Path) -> Dict[str, Any]:
                try:
                    file_size = file_path.stat().st_size

                    # 计算目标路径
                    if preserve_structure and recursive:
                        rel_path = file_path.relative_to(source_path)
                        target_file = target_path / rel_path
                        create_directory(target_file.parent)
                    else:
                        target_file = target_path / file_path.name

                    # 检查是否需要覆盖
                    if target_file.exists() and not overwrite:
                        # 生成唯一文件名
                        target_file = get_unique_filename(
                            target_file.parent, target_file.name
                        )

                    # 复制文件
                    if target_file.exists() and not overwrite:
                        return {
                            "success": False,
                            "action": "skipped",
                            "source_file": str(file_path),
                            "target_file": str(target_file),
                            "reason": "文件已存在且不允许覆盖",
                            "file_size": file_size,
                        }

                    copy_file_safe(file_path, target_file)

                    return {
                        "success": True,
                        "action": "copied",
                        "source_file": str(file_path),
                        "target_file": str(target_file),
                        "file_size": file_size,
                    }

                except Exception as e:
                    return {
                        "success": False,
                        "action": "failed",
                        "source_file": str(file_path),
                        "error": str(e),
                        "file_size": 0,
                    }

            # 批量处理
            copy_results = process_with_progress(
                files_to_copy, copy_single_file, "复制文件"
            )

            # 统计结果
            for copy_result in copy_results:
                if copy_result:
                    if copy_result["success"] and copy_result["action"] == "copied":
                        result["copied_files"].append(copy_result)
                        result["statistics"]["copied_count"] += 1
                        result["statistics"]["total_size"] += copy_result["file_size"]
                    elif copy_result["action"] == "skipped":
                        result["skipped_files"].append(copy_result)
                        result["statistics"]["skipped_count"] += 1
                    else:
                        result["failed_files"].append(copy_result)
                        result["statistics"]["failed_count"] += 1
                        result["success"] = False

            # 格式化大小信息
            result["statistics"]["total_size_formatted"] = format_file_size(
                result["statistics"]["total_size"]
            )

            self.logger.info(f"文件复制完成: {result['statistics']}")
            return result

        except Exception as e:
            raise ProcessingError(f"文件复制失败: {str(e)}")

    def move_files(
        self,
        source_dir: str,
        target_dir: str,
        file_patterns: Optional[List[str]] = None,
        recursive: bool = False,
        overwrite: bool = False,
        preserve_structure: bool = True,
    ) -> Dict[str, Any]:
        """批量移动文件

        Args:
            source_dir: 源目录
            target_dir: 目标目录
            file_patterns: 文件模式列表
            recursive: 是否递归处理
            overwrite: 是否覆盖已存在的文件
            preserve_structure: 是否保持目录结构

        Returns:
            Dict[str, Any]: 移动结果
        """
        try:
            source_path = validate_path(source_dir, must_exist=True, must_be_dir=True)
            target_path = validate_path(target_dir, must_exist=False)

            create_directory(target_path)

            self.logger.info(f"开始移动文件: {source_path} -> {target_path}")

            # 获取文件列表
            if file_patterns:
                files_to_move: List[Path] = []
                for pattern in file_patterns:
                    if recursive:
                        files_to_move.extend(source_path.rglob(pattern))
                    else:
                        files_to_move.extend(source_path.glob(pattern))
                files_to_move = [f for f in set(files_to_move) if f.is_file()]
            else:
                files_to_move = get_file_list(
                    source_path, extensions=None, recursive=recursive
                )

            result: Dict[str, Any] = {
                "success": True,
                "source_dir": str(source_path),
                "target_dir": str(target_path),
                "moved_files": [],
                "failed_files": [],
                "skipped_files": [],
                "statistics": {
                    "total_files": len(files_to_move),
                    "moved_count": 0,
                    "failed_count": 0,
                    "skipped_count": 0,
                    "total_size": 0,
                },
            }

            # 移动文件
            def move_single_file(file_path: Path) -> Dict[str, Any]:
                try:
                    file_size = file_path.stat().st_size

                    # 计算目标路径
                    if preserve_structure and recursive:
                        rel_path = file_path.relative_to(source_path)
                        target_file = target_path / rel_path
                        create_directory(target_file.parent)
                    else:
                        target_file = target_path / file_path.name

                    # 检查是否需要覆盖
                    if target_file.exists() and not overwrite:
                        target_file = get_unique_filename(
                            target_file.parent, target_file.name
                        )

                    # 移动文件
                    if target_file.exists() and not overwrite:
                        return {
                            "success": False,
                            "action": "skipped",
                            "source_file": str(file_path),
                            "target_file": str(target_file),
                            "reason": "文件已存在且不允许覆盖",
                            "file_size": file_size,
                        }

                    move_file_safe(file_path, target_file)

                    return {
                        "success": True,
                        "action": "moved",
                        "source_file": str(file_path),
                        "target_file": str(target_file),
                        "file_size": file_size,
                    }

                except Exception as e:
                    return {
                        "success": False,
                        "action": "failed",
                        "source_file": str(file_path),
                        "error": str(e),
                        "file_size": 0,
                    }

            # 批量处理
            move_results = process_with_progress(
                files_to_move, move_single_file, "移动文件"
            )

            # 统计结果
            for move_result in move_results:
                if move_result:
                    if move_result["success"] and move_result["action"] == "moved":
                        result["moved_files"].append(move_result)
                        result["statistics"]["moved_count"] += 1
                        result["statistics"]["total_size"] += move_result["file_size"]
                    elif move_result["action"] == "skipped":
                        result["skipped_files"].append(move_result)
                        result["statistics"]["skipped_count"] += 1
                    else:
                        result["failed_files"].append(move_result)
                        result["statistics"]["failed_count"] += 1
                        result["success"] = False

            # 格式化大小信息
            result["statistics"]["total_size_formatted"] = format_file_size(
                result["statistics"]["total_size"]
            )

            self.logger.info(f"文件移动完成: {result['statistics']}")
            return result

        except Exception as e:
            raise ProcessingError(f"文件移动失败: {str(e)}")

    def organize_by_extension(
        self,
        source_dir: str,
        output_dir: Optional[str] = None,
        copy_files: bool = False,
    ) -> Dict[str, Any]:
        """按扩展名组织文件（仅处理源目录下的一级文件）。"""
        try:
            source_path = validate_path(source_dir, must_exist=True, must_be_dir=True)
            target_path = (
                validate_path(output_dir, must_exist=False)
                if output_dir
                else source_path
            )
            create_directory(target_path)

            files_to_organize = [p for p in source_path.iterdir() if p.is_file()]

            result: Dict[str, Any] = {
                "success": True,
                "source_dir": str(source_path),
                "target_dir": str(target_path),
                "copy_files": copy_files,
                "organized_files": [],
                "failed_files": [],
                "statistics": {
                    "total_files": len(files_to_organize),
                    "copied_count": 0,
                    "moved_count": 0,
                    "failed_count": 0,
                },
            }

            def organize_single_file(file_path: Path) -> Dict[str, Any]:
                try:
                    suffix = file_path.suffix.lower().lstrip(".")
                    ext_folder = suffix if suffix else "no_extension"
                    ext_dir = target_path / ext_folder
                    create_directory(ext_dir)

                    target_file = ext_dir / file_path.name
                    if target_file.exists():
                        target_file = get_unique_filename(
                            target_file.parent, target_file.name
                        )

                    if copy_files:
                        copy_file_safe(file_path, target_file)
                        action = "copied"
                    else:
                        move_file_safe(file_path, target_file)
                        action = "moved"

                    return {
                        "success": True,
                        "action": action,
                        "source_file": str(file_path),
                        "target_file": str(target_file),
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "action": "failed",
                        "source_file": str(file_path),
                        "error": str(e),
                    }

            organize_results = process_with_progress(
                files_to_organize, organize_single_file, "按扩展名组织文件"
            )

            for organize_result in organize_results:
                if not organize_result:
                    continue
                if organize_result["success"]:
                    result["organized_files"].append(organize_result)
                    if organize_result["action"] == "copied":
                        result["statistics"]["copied_count"] += 1
                    else:
                        result["statistics"]["moved_count"] += 1
                else:
                    result["failed_files"].append(organize_result)
                    result["statistics"]["failed_count"] += 1
                    result["success"] = False

            self.logger.info(f"按扩展名组织完成: {result['statistics']}")
            return result

        except Exception as e:
            raise ProcessingError(f"按扩展名组织失败: {str(e)}")

    def delete_json_files_recursive(
        self,
        target_dir: str,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """递归扫描并删除目录中的 JSON 文件。"""
        try:
            target_path = validate_path(target_dir, must_exist=True, must_be_dir=True)
            json_files = [p for p in target_path.rglob("*.json") if p.is_file()]

            result: Dict[str, Any] = {
                "success": True,
                "target_dir": str(target_path),
                "dry_run": dry_run,
                "json_files": [str(p) for p in json_files],
                "failed_files": [],
                "statistics": {
                    "total_files": len(json_files),
                    "deleted_count": 0,
                    "failed_count": 0,
                },
            }

            if dry_run:
                return result

            for json_file in json_files:
                try:
                    json_file.unlink()
                    result["statistics"]["deleted_count"] += 1
                except Exception as e:
                    result["failed_files"].append(
                        {"file": str(json_file), "error": str(e)}
                    )
                    result["statistics"]["failed_count"] += 1
                    result["success"] = False

            return result

        except Exception as e:
            raise ProcessingError(f"递归删除JSON文件失败: {str(e)}")

    def move_images_by_count(
        self,
        source_dir: str,
        target_dir: str,
        count: int,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """按顺序移动指定数量的图片

        规则:
        - 源目录仅图片: 按文件名顺序移动
        - 源目录含子目录: 先处理源目录下图片，再按子目录名称顺序处理子目录图片
        """
        try:
            if count <= 0:
                raise ValueError("数量必须为正整数")

            source_path = validate_path(source_dir, must_exist=True, must_be_dir=True)
            target_path = validate_path(target_dir, must_exist=False)
            create_directory(target_path)

            image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

            root_images = sorted(
                [
                    p
                    for p in source_path.iterdir()
                    if p.is_file() and p.suffix.lower() in image_exts
                ],
                key=lambda p: p.name.lower(),
            )
            subdirs = sorted(
                [p for p in source_path.iterdir() if p.is_dir()],
                key=lambda p: p.name.lower(),
            )

            ordered_images: List[Path] = []
            ordered_images.extend(root_images)
            for subdir in subdirs:
                sub_images = sorted(
                    [
                        p
                        for p in subdir.iterdir()
                        if p.is_file() and p.suffix.lower() in image_exts
                    ],
                    key=lambda p: p.name.lower(),
                )
                ordered_images.extend(sub_images)

            result: Dict[str, Any] = {
                "success": True,
                "source_dir": str(source_path),
                "target_dir": str(target_path),
                "moved_files": [],
                "failed_files": [],
                "statistics": {
                    "total_candidates": len(ordered_images),
                    "requested_count": count,
                    "moved_count": 0,
                    "failed_count": 0,
                },
            }

            if count == 9999:
                count = len(ordered_images)

            moved = 0
            for image_path in ordered_images:
                if moved >= count:
                    break
                try:
                    target_file = target_path / image_path.name
                    if target_file.exists() and not overwrite:
                        target_file = get_unique_filename(
                            target_file.parent, target_file.name
                        )

                    move_file_safe(image_path, target_file)
                    result["moved_files"].append(str(target_file))
                    moved += 1
                except Exception as e:
                    result["failed_files"].append(
                        {"source_file": str(image_path), "error": str(e)}
                    )
                    result["statistics"]["failed_count"] += 1

            result["statistics"]["moved_count"] = moved
            return result

        except Exception as e:
            raise ProcessingError(f"按数量移动图片失败: {str(e)}")

    def rename_files_with_temp(
        self,
        target_dir: str,
        rename_pattern: str,
        file_patterns: Optional[List[str]] = None,
        recursive: bool = False,
        shuffle_order: bool = False,
        preview_only: bool = False,
    ) -> Dict[str, Any]:
        """使用临时重命名的批量重命名文件（避免冲突）

        Args:
            target_dir: 目标目录
            rename_pattern: 重命名模式（支持 {name}, {ext}, {index} 等占位符）
            file_patterns: 文件模式列表
            recursive: 是否递归处理
            shuffle_order: 是否打乱文件顺序
            preview_only: 仅预览，不实际重命名

        Returns:
            Dict[str, Any]: 重命名结果
        """
        import random
        import uuid

        try:
            target_path = validate_path(target_dir, must_exist=True, must_be_dir=True)

            self.logger.info(f"开始临时重命名文件: {target_path}")
            self.logger.info(f"重命名模式: {rename_pattern}")
            self.logger.info(f"打乱顺序: {shuffle_order}")

            # 获取文件列表
            if file_patterns:
                files_to_rename: List[Path] = []
                for pattern in file_patterns:
                    if recursive:
                        files_to_rename.extend(target_path.rglob(pattern))
                    else:
                        files_to_rename.extend(target_path.glob(pattern))
                files_to_rename = [f for f in set(files_to_rename) if f.is_file()]
            else:
                files_to_rename = get_file_list(
                    target_path, extensions=None, recursive=recursive
                )

            # 按名称排序以确保一致性
            files_to_rename.sort(key=lambda x: x.name)

            # 如果需要打乱顺序
            if shuffle_order:
                random.shuffle(files_to_rename)

            result: Dict[str, Any] = {
                "success": True,
                "target_dir": str(target_path),
                "rename_pattern": rename_pattern,
                "shuffle_order": shuffle_order,
                "preview_only": preview_only,
                "renamed_files": [],
                "failed_files": [],
                "statistics": {
                    "total_files": len(files_to_rename),
                    "renamed_count": 0,
                    "failed_count": 0,
                },
            }

            if preview_only:
                # 预览模式
                for index, file_path in enumerate(files_to_rename):
                    try:
                        new_name = rename_pattern.format(
                            name=file_path.stem,
                            ext=file_path.suffix,
                            index=index + 1,
                            index0=index,
                        )
                        new_path = self._build_safe_rename_target(
                            file_path.parent, new_name
                        )

                        result["renamed_files"].append(
                            {
                                "success": True,
                                "action": "preview",
                                "old_name": str(file_path),
                                "new_name": str(new_path),
                            }
                        )
                    except Exception as e:
                        result["failed_files"].append(
                            {
                                "success": False,
                                "action": "failed",
                                "old_name": str(file_path),
                                "error": str(e),
                            }
                        )
                        result["statistics"]["failed_count"] += 1
                        result["success"] = False

                return result

            # 第一阶段：临时重命名（避免冲突）
            temp_mappings: List[Dict[str, Any]] = []

            for file_path in files_to_rename:
                try:
                    # 生成临时文件名
                    temp_name = f"temp_{uuid.uuid4().hex[:8]}{file_path.suffix}"
                    temp_path = file_path.parent / temp_name

                    # 临时重命名
                    file_path.rename(temp_path)
                    temp_mappings.append(
                        {
                            "temp_path": temp_path,
                            "original_path": file_path,
                            "original_name": file_path.name,
                        }
                    )

                except Exception as e:
                    rollback_errors = []

                    # 第一阶段失败时，回滚此前已临时改名的文件
                    for mapping in reversed(temp_mappings):
                        try:
                            if (
                                mapping["temp_path"].exists()
                                and not mapping["original_path"].exists()
                            ):
                                mapping["temp_path"].rename(mapping["original_path"])
                        except Exception as rollback_error:
                            rollback_errors.append(
                                {
                                    "success": False,
                                    "action": "rollback_failed",
                                    "old_name": str(mapping["original_name"]),
                                    "error": str(rollback_error),
                                }
                            )

                    result["failed_files"].append(
                        {
                            "success": False,
                            "action": "temp_rename_failed",
                            "old_name": str(file_path),
                            "error": str(e),
                        }
                    )
                    result["failed_files"].extend(rollback_errors)
                    result["statistics"]["failed_count"] += 1 + len(rollback_errors)
                    result["success"] = False
                    return result

            # 第二阶段：正式重命名
            finalized_mappings: List[Dict[str, Any]] = []
            for index, mapping in enumerate(temp_mappings):
                temp_path = cast(Path, mapping["temp_path"])
                original_name = cast(str, mapping["original_name"])

                try:
                    # 生成最终文件名
                    new_name = rename_pattern.format(
                        name=Path(original_name).stem,
                        ext=temp_path.suffix,
                        index=index + 1,
                        index0=index,
                    )

                    final_path = self._build_safe_rename_target(
                        temp_path.parent, new_name
                    )

                    # 确保新文件名唯一
                    if final_path.exists():
                        final_path = get_unique_filename(
                            final_path.parent, final_path.name
                        )

                    # 最终重命名
                    temp_path.rename(final_path)
                    mapping["final_path"] = final_path
                    finalized_mappings.append(mapping)

                    result["renamed_files"].append(
                        {
                            "success": True,
                            "action": "renamed",
                            "old_name": original_name,
                            "new_name": str(final_path.name),
                        }
                    )
                    result["statistics"]["renamed_count"] += 1

                except Exception as e:
                    rollback_errors = []

                    # 1) 回滚已完成最终改名的文件
                    for done in reversed(finalized_mappings):
                        try:
                            final_done = done.get("final_path")
                            if (
                                final_done
                                and final_done.exists()
                                and not done["original_path"].exists()
                            ):
                                final_done.rename(done["original_path"])
                        except Exception as rollback_error:
                            rollback_errors.append(
                                {
                                    "success": False,
                                    "action": "rollback_failed",
                                    "old_name": str(done["original_name"]),
                                    "error": str(rollback_error),
                                }
                            )

                    # 2) 回滚当前及后续仍在 temp 状态的文件
                    pending_mappings = [mapping] + temp_mappings[index + 1 :]
                    for pending in reversed(pending_mappings):
                        try:
                            if (
                                pending["temp_path"].exists()
                                and not pending["original_path"].exists()
                            ):
                                pending["temp_path"].rename(pending["original_path"])
                        except Exception as rollback_error:
                            rollback_errors.append(
                                {
                                    "success": False,
                                    "action": "rollback_failed",
                                    "old_name": str(pending["original_name"]),
                                    "error": str(rollback_error),
                                }
                            )

                    result["failed_files"].append(
                        {
                            "success": False,
                            "action": "final_rename_failed",
                            "old_name": original_name,
                            "error": str(e),
                        }
                    )
                    result["failed_files"].extend(rollback_errors)
                    result["statistics"]["renamed_count"] = 0
                    result["renamed_files"] = []
                    result["statistics"]["failed_count"] += 1 + len(rollback_errors)
                    result["success"] = False
                    return result

            self.logger.info(f"临时重命名完成: {result['statistics']}")
            return result

        except Exception as e:
            raise ProcessingError(f"临时重命名失败: {str(e)}")

    def rename_images_labels_sync(
        self,
        images_dir: str,
        labels_dir: str,
        prefix: str,
        digits: int = 5,
        shuffle_order: bool = False,
    ) -> Dict[str, Any]:
        """同步重命名images和labels目录中的对应文件

        Args:
            images_dir: 图片目录路径
            labels_dir: 标签目录路径
            prefix: 文件名前缀
            digits: 数字位数
            shuffle_order: 是否打乱顺序

        Returns:
            Dict[str, Any]: 重命名结果
        """
        import random
        import uuid

        try:
            images_path = validate_path(images_dir, must_exist=True, must_be_dir=True)
            labels_path = validate_path(labels_dir, must_exist=True, must_be_dir=True)

            self.logger.info(f"开始同步重命名: {images_path} 和 {labels_path}")
            self.logger.info(
                f"前缀: {prefix}, 位数: {digits}, 打乱顺序: {shuffle_order}"
            )

            # 获取图片文件列表
            image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
            image_files: List[Path] = []
            for ext in image_extensions:
                image_files.extend(images_path.glob(f"*{ext}"))
                image_files.extend(images_path.glob(f"*{ext.upper()}"))

            # 去重处理（避免同一文件被匹配多次）
            image_files = list(set(image_files))

            # 记录找到的图片文件
            # self.logger.info(f"找到 {len(image_files)} 个图片文件:")
            # for img_file in image_files:
            #     self.logger.info(f"  图片文件: {img_file.name}")

            # 过滤出有对应标签文件的图片
            valid_pairs = []
            for img_file in image_files:
                label_file = labels_path / f"{img_file.stem}.txt"
                if label_file.exists():
                    valid_pairs.append((img_file, label_file))
                    # self.logger.info(f"  有效文件对: {img_file.name} <-> {label_file.name}")
                else:
                    self.logger.warning(
                        f"  图片文件 {img_file.name} 没有对应的标签文件 {img_file.stem}.txt"
                    )

            if not valid_pairs:
                raise ProcessingError("未找到有效的图片-标签文件对")

            # 按图片文件名排序
            valid_pairs.sort(key=lambda x: x[0].name)

            # 如果需要打乱顺序
            if shuffle_order:
                random.shuffle(valid_pairs)

            result: Dict[str, Any] = {
                "success": True,
                "images_dir": str(images_path),
                "labels_dir": str(labels_path),
                "prefix": prefix,
                "digits": digits,
                "shuffle_order": shuffle_order,
                "renamed_pairs": [],
                "failed_pairs": [],
                "statistics": {
                    "total_pairs": len(valid_pairs),
                    "renamed_count": 0,
                    "failed_count": 0,
                },
            }

            # 第一阶段：临时重命名所有文件
            temp_mappings: List[Dict[str, Any]] = []

            for img_file, label_file in valid_pairs:
                try:
                    # 检查文件是否真实存在
                    if not img_file.exists():
                        raise FileNotFoundError(f"图片文件不存在: {img_file}")
                    if not label_file.exists():
                        raise FileNotFoundError(f"标签文件不存在: {label_file}")

                    # 生成临时文件名
                    temp_id = uuid.uuid4().hex[:8]
                    temp_img_name = f"temp_img_{temp_id}{img_file.suffix}"
                    temp_label_name = f"temp_label_{temp_id}.txt"

                    temp_img_path = img_file.parent / temp_img_name
                    temp_label_path = label_file.parent / temp_label_name

                    # 临时重命名
                    img_renamed = False
                    label_renamed = False
                    try:
                        img_file.rename(temp_img_path)
                        img_renamed = True
                        label_file.rename(temp_label_path)
                        label_renamed = True
                    except Exception:
                        # 第一阶段失败时先回滚当前 pair
                        if (
                            img_renamed
                            and temp_img_path.exists()
                            and not img_file.exists()
                        ):
                            try:
                                temp_img_path.rename(img_file)
                            except Exception:
                                pass
                        if (
                            label_renamed
                            and temp_label_path.exists()
                            and not label_file.exists()
                        ):
                            try:
                                temp_label_path.rename(label_file)
                            except Exception:
                                pass
                        raise

                    temp_mappings.append(
                        {
                            "temp_img": temp_img_path,
                            "temp_label": temp_label_path,
                            "original_img": img_file,
                            "original_label": label_file,
                            "original_img_name": img_file.name,
                            "original_label_name": label_file.name,
                            "img_ext": img_file.suffix,
                        }
                    )

                except Exception as e:
                    rollback_errors = []

                    # 第一阶段失败时，回滚此前所有已进入 temp 的 pair
                    for done in reversed(temp_mappings):
                        try:
                            if (
                                done["temp_img"].exists()
                                and not done["original_img"].exists()
                            ):
                                done["temp_img"].rename(done["original_img"])
                        except Exception as rollback_error:
                            rollback_errors.append(
                                {
                                    "success": False,
                                    "action": "rollback_failed",
                                    "img_file": str(done["original_img_name"]),
                                    "label_file": str(done["original_label_name"]),
                                    "error": str(rollback_error),
                                }
                            )
                        try:
                            if (
                                done["temp_label"].exists()
                                and not done["original_label"].exists()
                            ):
                                done["temp_label"].rename(done["original_label"])
                        except Exception as rollback_error:
                            rollback_errors.append(
                                {
                                    "success": False,
                                    "action": "rollback_failed",
                                    "img_file": str(done["original_img_name"]),
                                    "label_file": str(done["original_label_name"]),
                                    "error": str(rollback_error),
                                }
                            )

                    result["failed_pairs"].append(
                        {
                            "success": False,
                            "action": "temp_rename_failed",
                            "img_file": str(img_file),
                            "label_file": str(label_file),
                            "error": str(e),
                        }
                    )
                    result["failed_pairs"].extend(rollback_errors)
                    result["statistics"]["failed_count"] += 1 + len(rollback_errors)
                    result["success"] = False
                    return result

            # 第二阶段：正式重命名
            finalized_mappings: List[Dict[str, Any]] = []
            for index, mapping in enumerate(temp_mappings):
                final_img_path = None
                final_label_path = None
                try:
                    # 生成最终文件名
                    if prefix:
                        final_name = f"{prefix}_{index+1:0{digits}d}"
                    else:
                        final_name = f"{index+1:0{digits}d}"

                    # 生成不冲突且同 stem 的最终文件名
                    stem_candidate = final_name
                    suffix_candidate = mapping["img_ext"]
                    counter = 1
                    while True:
                        img_candidate = self._build_safe_rename_target(
                            mapping["temp_img"].parent,
                            f"{stem_candidate}{suffix_candidate}",
                        )
                        label_candidate = self._build_safe_rename_target(
                            mapping["temp_label"].parent,
                            f"{stem_candidate}.txt",
                        )
                        if not img_candidate.exists() and not label_candidate.exists():
                            final_img_path = img_candidate
                            final_label_path = label_candidate
                            break
                        stem_candidate = f"{final_name}_{counter}"
                        counter += 1

                    # 最终重命名
                    mapping["temp_img"].rename(final_img_path)
                    mapping["temp_label"].rename(final_label_path)
                    mapping["final_img"] = final_img_path
                    mapping["final_label"] = final_label_path
                    finalized_mappings.append(mapping)

                    result["renamed_pairs"].append(
                        {
                            "success": True,
                            "action": "renamed",
                            "old_img": mapping["original_img_name"],
                            "old_label": mapping["original_label_name"],
                            "new_img": final_img_path.name,
                            "new_label": final_label_path.name,
                        }
                    )
                    result["statistics"]["renamed_count"] += 1

                except Exception as e:
                    rollback_errors = []

                    # 0) 先回滚当前 pair（可能已从 temp_img 改到 final_img）
                    try:
                        if (
                            final_img_path
                            and final_img_path.exists()
                            and not mapping["original_img"].exists()
                        ):
                            final_img_path.rename(mapping["original_img"])
                        elif (
                            mapping["temp_img"].exists()
                            and not mapping["original_img"].exists()
                        ):
                            mapping["temp_img"].rename(mapping["original_img"])
                    except Exception as rollback_error:
                        rollback_errors.append(
                            {
                                "success": False,
                                "action": "rollback_failed",
                                "img_file": str(mapping["original_img_name"]),
                                "label_file": str(mapping["original_label_name"]),
                                "error": str(rollback_error),
                            }
                        )

                    try:
                        if (
                            final_label_path
                            and final_label_path.exists()
                            and not mapping["original_label"].exists()
                        ):
                            final_label_path.rename(mapping["original_label"])
                        elif (
                            mapping["temp_label"].exists()
                            and not mapping["original_label"].exists()
                        ):
                            mapping["temp_label"].rename(mapping["original_label"])
                    except Exception as rollback_error:
                        rollback_errors.append(
                            {
                                "success": False,
                                "action": "rollback_failed",
                                "img_file": str(mapping["original_img_name"]),
                                "label_file": str(mapping["original_label_name"]),
                                "error": str(rollback_error),
                            }
                        )

                    # 1) 回滚已完成最终改名的 pair
                    for done in reversed(finalized_mappings):
                        try:
                            final_img = done.get("final_img")
                            if (
                                final_img
                                and final_img.exists()
                                and not done["original_img"].exists()
                            ):
                                final_img.rename(done["original_img"])
                        except Exception as rollback_error:
                            rollback_errors.append(
                                {
                                    "success": False,
                                    "action": "rollback_failed",
                                    "img_file": str(done["original_img_name"]),
                                    "label_file": str(done["original_label_name"]),
                                    "error": str(rollback_error),
                                }
                            )
                        try:
                            final_label = done.get("final_label")
                            if (
                                final_label
                                and final_label.exists()
                                and not done["original_label"].exists()
                            ):
                                final_label.rename(done["original_label"])
                        except Exception as rollback_error:
                            rollback_errors.append(
                                {
                                    "success": False,
                                    "action": "rollback_failed",
                                    "img_file": str(done["original_img_name"]),
                                    "label_file": str(done["original_label_name"]),
                                    "error": str(rollback_error),
                                }
                            )

                    # 2) 回滚后续仍在 temp 状态的 pair
                    pending_mappings = temp_mappings[index + 1 :]
                    for pending in reversed(pending_mappings):
                        try:
                            if (
                                pending["temp_img"].exists()
                                and not pending["original_img"].exists()
                            ):
                                pending["temp_img"].rename(pending["original_img"])
                        except Exception as rollback_error:
                            rollback_errors.append(
                                {
                                    "success": False,
                                    "action": "rollback_failed",
                                    "img_file": str(pending["original_img_name"]),
                                    "label_file": str(pending["original_label_name"]),
                                    "error": str(rollback_error),
                                }
                            )
                        try:
                            if (
                                pending["temp_label"].exists()
                                and not pending["original_label"].exists()
                            ):
                                pending["temp_label"].rename(pending["original_label"])
                        except Exception as rollback_error:
                            rollback_errors.append(
                                {
                                    "success": False,
                                    "action": "rollback_failed",
                                    "img_file": str(pending["original_img_name"]),
                                    "label_file": str(pending["original_label_name"]),
                                    "error": str(rollback_error),
                                }
                            )

                    result["failed_pairs"].append(
                        {
                            "success": False,
                            "action": "final_rename_failed",
                            "img_file": mapping["original_img_name"],
                            "label_file": mapping["original_label_name"],
                            "error": str(e),
                        }
                    )
                    result["failed_pairs"].extend(rollback_errors)
                    result["statistics"]["renamed_count"] = 0
                    result["renamed_pairs"] = []
                    result["statistics"]["failed_count"] += 1 + len(rollback_errors)
                    result["success"] = False
                    return result

            # 记录详细的失败信息到日志
            if result["failed_pairs"]:
                self.logger.warning(
                    f"有 {len(result['failed_pairs'])} 个文件对重命名失败:"
                )
                for failed_item in result["failed_pairs"]:
                    self.logger.warning(
                        f"  失败文件: {failed_item.get('img_file', 'N/A')} / {failed_item.get('label_file', 'N/A')}"
                    )
                    self.logger.warning(
                        f"  失败原因: {failed_item.get('error', 'N/A')}"
                    )
                    self.logger.warning(
                        f"  失败阶段: {failed_item.get('action', 'N/A')}"
                    )

            self.logger.info(f"同步重命名完成: {result['statistics']}")
            return result

        except Exception as e:
            raise ProcessingError(f"同步重命名失败: {str(e)}")
