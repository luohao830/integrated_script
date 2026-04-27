#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
label_processor.py

标签处理器

提供标签文件的创建、修改、转换等功能，特别针对机器学习标注数据。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from ...config.exceptions import ProcessingError, ValidationError
from ...core.base import BaseProcessor
from ...core.progress import process_with_progress
from ...core.utils import (
    create_directory,
    delete_file_safe,
    get_file_list,
    validate_path,
)


class LabelProcessor(BaseProcessor):
    """标签处理器

    提供标签文件的创建、修改、转换等功能。

    Attributes:
        supported_formats (List[str]): 支持的标签格式
        image_extensions (List[str]): 支持的图像扩展名
    """

    def __init__(self, **kwargs):
        """初始化标签处理器"""
        # 先设置属性，再调用父类初始化
        self.supported_formats = ["yolo", "coco", "pascal_voc"]
        self.image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]

        # 调用父类初始化
        super().__init__(name="LabelProcessor", **kwargs)

    def initialize(self) -> None:
        """初始化处理器"""
        self.logger.info("标签处理器初始化完成")
        self.logger.debug(f"支持的格式: {self.supported_formats}")
        self.logger.debug(f"图像扩展名: {self.image_extensions}")

    def process(self, *args, **kwargs) -> Any:
        """主要处理方法（由子方法实现具体功能）"""
        raise NotImplementedError("请使用具体的处理方法")

    def create_empty_labels(
        self, images_dir: str, labels_dir: Optional[str] = None, overwrite: bool = False
    ) -> Dict[str, Any]:
        """为图像创建空白标签文件

        Args:
            images_dir: 图像目录
            labels_dir: 标签目录（如果为None，则在图像目录创建）
            overwrite: 是否覆盖已存在的标签文件


        Returns:
            Dict[str, Any]: 创建结果
        """
        try:
            images_path = validate_path(images_dir, must_exist=True, must_be_dir=True)

            if labels_dir is None:
                labels_path = images_path
            else:
                labels_path = validate_path(labels_dir, must_exist=False)
                # 确保标签目录存在，如果不存在则创建
                create_directory(labels_path)

            self.logger.info(f"开始创建空白标签: {images_path} -> {labels_path}")

            # 获取图像文件
            image_files = get_file_list(
                images_path, self.image_extensions, recursive=False
            )

            result: Dict[str, Any] = {
                "success": True,
                "images_dir": str(images_path),
                "labels_dir": str(labels_path),
                "created_labels": [],
                "skipped_labels": [],
                "failed_labels": [],
                "statistics": {
                    "total_images": len(image_files),
                    "created_count": 0,
                    "skipped_count": 0,
                    "failed_count": 0,
                },
            }

            # 创建空白标签文件
            def create_empty_label(img_file: Path) -> Dict[str, Any]:
                try:
                    label_file = labels_path / f"{img_file.stem}.txt"

                    if label_file.exists() and not overwrite:
                        return {
                            "success": False,
                            "action": "skipped",
                            "image_file": str(img_file),
                            "label_file": str(label_file),
                            "reason": "标签文件已存在",
                        }

                    # 创建空白标签文件
                    with open(label_file, "w", encoding="utf-8") as f:
                        f.write("")  # 空文件

                    return {
                        "success": True,
                        "action": "created",
                        "image_file": str(img_file),
                        "label_file": str(label_file),
                    }

                except Exception as e:
                    return {
                        "success": False,
                        "action": "failed",
                        "image_file": str(img_file),
                        "error": str(e),
                    }

            # 批量处理
            creation_results = process_with_progress(
                image_files, create_empty_label, "创建空白标签"
            )

            # 统计结果
            for creation_result in creation_results:
                if creation_result:
                    if (
                        creation_result["success"]
                        and creation_result["action"] == "created"
                    ):
                        result["created_labels"].append(creation_result)
                        result["statistics"]["created_count"] += 1
                    elif creation_result["action"] == "skipped":
                        result["skipped_labels"].append(creation_result)
                        result["statistics"]["skipped_count"] += 1
                    else:
                        result["failed_labels"].append(creation_result)
                        result["statistics"]["failed_count"] += 1
                        result["success"] = False

            self.logger.info(f"空白标签创建完成: {result['statistics']}")
            return result

        except Exception as e:
            raise ProcessingError(f"创建空白标签失败: {str(e)}")

    def flip_labels(
        self, labels_dir: str, flip_type: str = "horizontal", backup: bool = True
    ) -> Dict[str, Any]:
        """翻转标签坐标

        Args:
            labels_dir: 标签目录
            flip_type: 翻转类型 ('horizontal', 'vertical', 'both')
            backup: 是否备份原文件

        Returns:
            Dict[str, Any]: 翻转结果
        """
        try:
            labels_path = validate_path(labels_dir, must_exist=True, must_be_dir=True)

            if flip_type not in ["horizontal", "vertical", "both"]:
                raise ValidationError(f"不支持的翻转类型: {flip_type}")

            self.logger.info(f"开始翻转标签: {labels_path}")
            self.logger.info(f"翻转类型: {flip_type}")

            # 获取标签文件
            label_files = get_file_list(labels_path, [".txt"], recursive=False)

            result: Dict[str, Any] = {
                "success": True,
                "labels_dir": str(labels_path),
                "flip_type": flip_type,
                "backup": backup,
                "flipped_labels": [],
                "failed_labels": [],
                "statistics": {
                    "total_labels": len(label_files),
                    "flipped_count": 0,
                    "failed_count": 0,
                },
            }

            # 翻转单个标签文件
            def flip_single_label(label_file: Path) -> Dict[str, Any]:
                try:
                    # 备份原文件
                    if backup:
                        backup_file = label_file.with_suffix(".txt.bak")
                        import shutil

                        shutil.copy2(label_file, backup_file)

                    # 读取标签内容
                    with open(label_file, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                    flipped_lines = []
                    for line in lines:
                        line = line.strip()
                        if not line:
                            flipped_lines.append("")
                            continue

                        parts = line.split()
                        if len(parts) < 5:
                            flipped_lines.append(line)  # 保持原样
                            continue

                        class_id = parts[0]
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                        # 翻转坐标
                        if flip_type in ["horizontal", "both"]:
                            x_center = 1.0 - x_center

                        if flip_type in ["vertical", "both"]:
                            y_center = 1.0 - y_center

                        # 重建行
                        new_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                        if len(parts) > 5:  # 如果有额外的参数
                            new_line += " " + " ".join(parts[5:])

                        flipped_lines.append(new_line)

                    # 写回文件
                    with open(label_file, "w", encoding="utf-8") as f:
                        f.write("\n".join(flipped_lines))

                    return {
                        "success": True,
                        "label_file": str(label_file),
                        "backup_file": str(backup_file) if backup else None,
                    }

                except Exception as e:
                    return {
                        "success": False,
                        "label_file": str(label_file),
                        "error": str(e),
                    }

            # 批量处理
            flip_results = process_with_progress(
                label_files, flip_single_label, f"翻转标签 ({flip_type})"
            )

            # 统计结果
            for flip_result in flip_results:
                if flip_result:
                    if flip_result["success"]:
                        result["flipped_labels"].append(flip_result)
                        result["statistics"]["flipped_count"] += 1
                    else:
                        result["failed_labels"].append(flip_result)
                        result["statistics"]["failed_count"] += 1
                        result["success"] = False

            self.logger.info(f"标签翻转完成: {result['statistics']}")
            return result

        except Exception as e:
            raise ProcessingError(f"标签翻转失败: {str(e)}")

    def filter_labels_by_class(
        self,
        labels_dir: str,
        target_classes: List[int],
        action: str = "keep",
        backup: bool = True,
    ) -> Dict[str, Any]:
        """根据类别过滤标签

        Args:
            labels_dir: 标签目录
            target_classes: 目标类别列表
            action: 操作类型 ('keep' 保留, 'remove' 移除)
            backup: 是否备份原文件

        Returns:
            Dict[str, Any]: 过滤结果
        """
        try:
            labels_path = validate_path(labels_dir, must_exist=True, must_be_dir=True)

            if action not in ["keep", "remove"]:
                raise ValidationError(f"不支持的操作类型: {action}")

            self.logger.info(f"开始过滤标签: {labels_path}")
            self.logger.info(f"目标类别: {target_classes}, 操作: {action}")

            # 获取标签文件
            label_files = get_file_list(labels_path, [".txt"], recursive=False)

            result: Dict[str, Any] = {
                "success": True,
                "labels_dir": str(labels_path),
                "target_classes": target_classes,
                "action": action,
                "backup": backup,
                "processed_labels": [],
                "failed_labels": [],
                "statistics": {
                    "total_labels": len(label_files),
                    "processed_count": 0,
                    "failed_count": 0,
                    "annotations_removed": 0,
                    "annotations_kept": 0,
                },
            }

            # 过滤单个标签文件
            def filter_single_label(label_file: Path) -> Dict[str, Any]:
                try:
                    # 备份原文件
                    if backup:
                        backup_file = label_file.with_suffix(".txt.bak")
                        import shutil

                        shutil.copy2(label_file, backup_file)

                    # 读取标签内容
                    with open(label_file, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                    filtered_lines = []
                    annotations_removed = 0
                    annotations_kept = 0

                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue

                        parts = line.split()
                        if len(parts) < 5:
                            continue

                        try:
                            class_id = int(parts[0])

                            if action == "keep":
                                if class_id in target_classes:
                                    filtered_lines.append(line)
                                    annotations_kept += 1
                                else:
                                    annotations_removed += 1
                            else:  # action == 'remove'
                                if class_id not in target_classes:
                                    filtered_lines.append(line)
                                    annotations_kept += 1
                                else:
                                    annotations_removed += 1

                        except ValueError:
                            # 如果类别ID不是整数，保持原样
                            filtered_lines.append(line)
                            annotations_kept += 1

                    # 写回文件
                    with open(label_file, "w", encoding="utf-8") as f:
                        f.write("\n".join(filtered_lines))
                        if filtered_lines:  # 如果有内容，添加最后的换行符
                            f.write("\n")

                    return {
                        "success": True,
                        "label_file": str(label_file),
                        "backup_file": str(backup_file) if backup else None,
                        "annotations_removed": annotations_removed,
                        "annotations_kept": annotations_kept,
                    }

                except Exception as e:
                    return {
                        "success": False,
                        "label_file": str(label_file),
                        "error": str(e),
                    }

            # 批量处理
            filter_results = process_with_progress(
                label_files, filter_single_label, f"过滤标签 ({action})"
            )

            # 统计结果
            for filter_result in filter_results:
                if filter_result:
                    if filter_result["success"]:
                        result["processed_labels"].append(filter_result)
                        result["statistics"]["processed_count"] += 1
                        result["statistics"]["annotations_removed"] += filter_result[
                            "annotations_removed"
                        ]
                        result["statistics"]["annotations_kept"] += filter_result[
                            "annotations_kept"
                        ]
                    else:
                        result["failed_labels"].append(filter_result)
                        result["statistics"]["failed_count"] += 1
                        result["success"] = False

            self.logger.info(f"标签过滤完成: {result['statistics']}")
            return result

        except Exception as e:
            raise ProcessingError(f"标签过滤失败: {str(e)}")

    def remove_empty_labels_and_images(
        self,
        dataset_dir: str,
        images_subdir: str = "images",
        labels_subdir: str = "labels",
    ) -> Dict[str, Any]:
        """删除空标签文件及其对应的图像

        Args:
            dataset_dir: 数据集目录
            images_subdir: 图像子目录名
            labels_subdir: 标签子目录名

        Returns:
            Dict[str, Any]: 删除结果
        """
        try:
            dataset_path = validate_path(dataset_dir, must_exist=True, must_be_dir=True)

            images_path = dataset_path / images_subdir
            labels_path = dataset_path / labels_subdir

            if not images_path.exists():
                raise ValidationError(f"图像目录不存在: {images_path}")
            if not labels_path.exists():
                raise ValidationError(f"标签目录不存在: {labels_path}")

            self.logger.info(f"开始删除空标签及对应图像: {dataset_path}")

            # 获取标签文件
            label_files = get_file_list(labels_path, [".txt"], recursive=False)

            result: Dict[str, Any] = {
                "success": True,
                "dataset_dir": str(dataset_path),
                "images_dir": str(images_path),
                "labels_dir": str(labels_path),
                "removed_pairs": [],
                "failed_operations": [],
                "statistics": {
                    "total_labels": len(label_files),
                    "empty_labels": 0,
                    "removed_images": 0,
                    "removed_labels": 0,
                    "failed_count": 0,
                },
            }

            # 检查并删除空标签
            def process_label_file(label_file: Path) -> Dict[str, Any]:
                try:
                    # 检查标签文件是否为空
                    with open(label_file, "r", encoding="utf-8") as f:
                        content = f.read().strip()

                    if not content:
                        # 查找对应的图像文件
                        image_files_found = []

                        # 首先检查没有扩展名的文件
                        img_file_no_ext = images_path / label_file.stem
                        if img_file_no_ext.exists():
                            image_files_found.append(img_file_no_ext)

                        # 然后检查有扩展名的文件
                        for ext in self.image_extensions:
                            img_file = images_path / f"{label_file.stem}{ext}"
                            if img_file.exists():
                                image_files_found.append(img_file)

                        # 删除标签文件
                        delete_file_safe(label_file)

                        # 删除对应的图像文件
                        for img_file in image_files_found:
                            delete_file_safe(img_file)

                        return {
                            "success": True,
                            "action": "removed",
                            "label_file": str(label_file),
                            "image_files": [str(f) for f in image_files_found],
                            "removed_images_count": len(image_files_found),
                        }
                    else:
                        return {
                            "success": True,
                            "action": "kept",
                            "label_file": str(label_file),
                        }

                except Exception as e:
                    return {
                        "success": False,
                        "action": "failed",
                        "label_file": str(label_file),
                        "error": str(e),
                    }

            # 批量处理
            process_results = process_with_progress(
                label_files, process_label_file, "删除空标签及图像"
            )

            # 统计结果
            for process_result in process_results:
                if process_result:
                    if process_result["success"]:
                        if process_result["action"] == "removed":
                            result["removed_pairs"].append(process_result)
                            result["statistics"]["empty_labels"] += 1
                            result["statistics"]["removed_labels"] += 1
                            result["statistics"]["removed_images"] += process_result[
                                "removed_images_count"
                            ]
                    else:
                        result["failed_operations"].append(process_result)
                        result["statistics"]["failed_count"] += 1
                        result["success"] = False

            self.logger.info(f"空标签删除完成: {result['statistics']}")
            return result

        except Exception as e:
            raise ProcessingError(f"删除空标签失败: {str(e)}")

    def remove_labels_with_only_class(
        self,
        dataset_dir: str,
        target_class: int,
        images_subdir: str = "images",
        labels_subdir: str = "labels",
    ) -> Dict[str, Any]:
        """删除只包含指定类别的标签文件及其对应图像

        Args:
            dataset_dir: 数据集目录
            target_class: 目标类别
            images_subdir: 图像子目录名
            labels_subdir: 标签子目录名

        Returns:
            Dict[str, Any]: 删除结果
        """
        try:
            dataset_path = validate_path(dataset_dir, must_exist=True, must_be_dir=True)

            images_path = dataset_path / images_subdir
            labels_path = dataset_path / labels_subdir

            if not images_path.exists():
                raise ValidationError(f"图像目录不存在: {images_path}")
            if not labels_path.exists():
                raise ValidationError(f"标签目录不存在: {labels_path}")

            self.logger.info(
                f"开始删除只包含类别{target_class}的标签及图像: {dataset_path}"
            )

            # 获取标签文件
            label_files = [
                label_file
                for label_file in get_file_list(labels_path, [".txt"], recursive=False)
                if label_file.name != "classes.txt"
            ]

            result: Dict[str, Any] = {
                "success": True,
                "dataset_dir": str(dataset_path),
                "images_dir": str(images_path),
                "labels_dir": str(labels_path),
                "target_class": target_class,
                "removed_pairs": [],
                "failed_operations": [],
                "statistics": {
                    "total_labels": len(label_files),
                    "target_class_only_labels": 0,
                    "removed_images": 0,
                    "removed_labels": 0,
                    "failed_count": 0,
                },
            }

            # 检查并删除只包含目标类别的标签
            def process_label_file(label_file: Path) -> Dict[str, Any]:
                try:
                    # 读取标签文件
                    with open(label_file, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                    classes_in_file = set()
                    valid_annotations = 0

                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue

                        parts = line.split()
                        if len(parts) >= 5:
                            try:
                                class_id = int(parts[0])
                                classes_in_file.add(class_id)
                                valid_annotations += 1
                            except ValueError:
                                continue

                    # 检查是否只包含目标类别
                    if valid_annotations > 0 and classes_in_file == {target_class}:
                        # 查找对应的图像文件
                        image_files_found = []

                        # 首先检查没有扩展名的文件
                        img_file_no_ext = images_path / label_file.stem
                        if img_file_no_ext.exists():
                            image_files_found.append(img_file_no_ext)

                        # 然后检查有扩展名的文件
                        for ext in self.image_extensions:
                            img_file = images_path / f"{label_file.stem}{ext}"
                            if img_file.exists():
                                image_files_found.append(img_file)

                        # 删除标签文件
                        delete_file_safe(label_file)

                        # 删除对应的图像文件
                        for img_file in image_files_found:
                            delete_file_safe(img_file)

                        return {
                            "success": True,
                            "action": "removed",
                            "label_file": str(label_file),
                            "image_files": [str(f) for f in image_files_found],
                            "removed_images_count": len(image_files_found),
                            "classes_in_file": list(classes_in_file),
                        }
                    else:
                        return {
                            "success": True,
                            "action": "kept",
                            "label_file": str(label_file),
                            "classes_in_file": list(classes_in_file),
                        }

                except Exception as e:
                    return {
                        "success": False,
                        "action": "failed",
                        "label_file": str(label_file),
                        "error": str(e),
                    }

            # 批量处理
            process_results = process_with_progress(
                label_files, process_label_file, f"删除只包含类别{target_class}的标签"
            )

            # 统计结果
            for process_result in process_results:
                if process_result:
                    if process_result["success"]:
                        if process_result["action"] == "removed":
                            result["removed_pairs"].append(process_result)
                            result["statistics"]["target_class_only_labels"] += 1
                            result["statistics"]["removed_labels"] += 1
                            result["statistics"]["removed_images"] += process_result[
                                "removed_images_count"
                            ]
                    else:
                        result["failed_operations"].append(process_result)
                        result["statistics"]["failed_count"] += 1
                        result["success"] = False

            self.logger.info(f"类别{target_class}标签删除完成: {result['statistics']}")
            return result

        except Exception as e:
            raise ProcessingError(f"删除指定类别标签失败: {str(e)}")
