#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset_processor.py

数据集处理器

提供数据集验证、清理、转换等功能，特别针对机器学习数据集。
"""

from pathlib import Path
from typing import Any, Dict, List, cast

from ..core.base import BaseProcessor
from ..core.progress import process_with_progress
from ..core.utils import get_file_list


class DatasetProcessor(BaseProcessor):
    """数据集处理器

    提供数据集验证、清理、转换等功能，特别针对机器学习数据集。

    Attributes:
        supported_formats (List[str]): 支持的数据集格式
        image_extensions (List[str]): 支持的图像扩展名
        annotation_extensions (List[str]): 支持的标注文件扩展名
    """

    def __init__(self, **kwargs):
        """初始化数据集处理器"""
        # 先设置属性，再调用父类初始化
        self.supported_formats = ["yolo", "coco", "pascal_voc", "csv"]
        self.image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
        self.annotation_extensions = [".txt", ".xml", ".json", ".csv"]

        # 调用父类初始化
        super().__init__(**kwargs)

    def initialize(self) -> None:
        """初始化处理器"""
        self.logger.info("数据集处理器初始化完成")
        self.logger.debug(f"支持的格式: {self.supported_formats}")
        self.logger.debug(f"图像扩展名: {self.image_extensions}")
        self.logger.debug(f"标注扩展名: {self.annotation_extensions}")

    def process(self, *args, **kwargs) -> Any:
        """主要处理方法（由子方法实现具体功能）"""
        raise NotImplementedError("请使用具体的处理方法")

    def _validate_yolo_dataset(
        self, dataset_dir: Path, check_integrity: bool
    ) -> Dict[str, Any]:
        """验证YOLO格式数据集

        只检查images和labels目录中的文件是否一一匹配，忽略其他文件。
        """
        result: Dict[str, Any] = {
            "success": True,
            "dataset_path": str(dataset_dir),
            "format": "yolo",
            "issues": [],
            "statistics": {
                "total_images": 0,
                "total_labels": 0,
                "orphaned_images": 0,
                "orphaned_labels": 0,
                "invalid_labels": 0,
                "empty_labels": 0,
            },
        }

        # 获取图像和标签文件
        # 只检查标准YOLO目录结构（images和labels目录）
        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"

        if not images_dir.exists():
            self.logger.warning(f"images目录不存在: {images_dir}")
            return result

        if not labels_dir.exists():
            self.logger.warning(f"labels目录不存在: {labels_dir}")
            return result

        # 只从images和labels目录获取文件，不递归搜索其他目录
        image_files = get_file_list(images_dir, self.image_extensions, recursive=False)
        label_files = get_file_list(labels_dir, [".txt"], recursive=False)

        stats = cast(Dict[str, Any], result["statistics"])
        issues = cast(List[Dict[str, Any]], result["issues"])

        stats["total_images"] = len(image_files)
        stats["total_labels"] = len(label_files)

        # 创建文件映射
        image_stems = {f.stem: f for f in image_files}
        label_stems = {f.stem: f for f in label_files}

        # 调试日志：打印文件映射信息
        self.logger.info(f"图像文件stems: {list(image_stems.keys())}")
        self.logger.info(f"标签文件stems: {list(label_stems.keys())}")

        # 检查孤立文件
        orphaned_images: List[Path] = []
        orphaned_labels: List[Path] = []

        for stem, img_file in image_stems.items():
            if stem not in label_stems:
                orphaned_images.append(img_file)
                self.logger.info(f"发现孤立图像: {img_file} (stem: {stem})")

        for stem, label_file in label_stems.items():
            if stem not in image_stems:
                orphaned_labels.append(label_file)
                self.logger.info(f"发现孤立标签: {label_file} (stem: {stem})")

        stats["orphaned_images"] = len(orphaned_images)
        stats["orphaned_labels"] = len(orphaned_labels)

        # 计算成功配对的数量
        matched_pairs = len(set(image_stems.keys()) & set(label_stems.keys()))
        stats["matched_pairs"] = matched_pairs

        # 调试日志：打印孤立文件统计
        self.logger.info(
            f"孤立图像数量: {len(orphaned_images)}, 孤立标签数量: {len(orphaned_labels)}, 成功配对数量: {matched_pairs}"
        )

        if orphaned_images:
            issue = {
                "type": "orphaned_images",
                "count": len(orphaned_images),
                "files": [str(f) for f in orphaned_images],  # 包含所有文件用于清理
                "preview": [
                    str(f) for f in orphaned_images[:10]
                ],  # 只显示前10个用于预览
            }
            issues.append(issue)
            self.logger.info(
                f"添加孤立图像issue到结果: {issue['type']}, count={issue['count']}"
            )

        if orphaned_labels:
            issue = {
                "type": "orphaned_labels",
                "count": len(orphaned_labels),
                "files": [str(f) for f in orphaned_labels],  # 包含所有文件用于清理
                "preview": [
                    str(f) for f in orphaned_labels[:10]
                ],  # 只显示前10个用于预览
            }
            issues.append(issue)
            self.logger.info(
                f"添加孤立标签issue到结果: {issue['type']}, count={issue['count']}"
            )

        # 检查标签文件完整性
        if check_integrity:
            invalid_labels: List[Dict[str, Any]] = []
            empty_labels: List[Dict[str, Any]] = []

            def validate_label_file(label_file: Path) -> Dict[str, Any]:
                try:
                    with open(label_file, "r", encoding="utf-8") as f:
                        content = f.read().strip()

                    if not content:
                        return {"type": "empty", "file": label_file}

                    lines = content.split("\n")
                    for line_num, line in enumerate(lines, 1):
                        line = line.strip()
                        if not line:
                            continue

                        parts = line.split()
                        if len(parts) < 5:
                            return {
                                "type": "invalid",
                                "file": label_file,
                                "reason": f"第{line_num}行格式错误: {line}",
                            }

                        try:
                            class_id = int(parts[0])

                            # 检查类别ID
                            if class_id < 0:
                                return {
                                    "type": "invalid",
                                    "file": label_file,
                                    "reason": f"第{line_num}行类别ID不能为负数: {line}",
                                }

                            # 检查坐标值范围和有效性
                            for i, coord_str in enumerate(parts[1:5]):
                                try:
                                    coord = float(coord_str)
                                    # 检查是否为负数
                                    if coord < 0.0:
                                        return {
                                            "type": "invalid",
                                            "file": label_file,
                                            "reason": f"第{line_num}行第{i+2}列坐标不能为负数: {coord} (行内容: {line})",
                                        }
                                    # 检查是否超出范围[0,1]
                                    if coord > 1.0:
                                        return {
                                            "type": "invalid",
                                            "file": label_file,
                                            "reason": f"第{line_num}行第{i+2}列坐标超出范围[0,1]: {coord} (行内容: {line})",
                                        }
                                except ValueError:
                                    return {
                                        "type": "invalid",
                                        "file": label_file,
                                        "reason": f"第{line_num}行第{i+2}列坐标格式错误: {coord_str} (行内容: {line})",
                                    }

                        except ValueError:
                            return {
                                "type": "invalid",
                                "file": label_file,
                                "reason": f"第{line_num}行类别ID格式错误: {line}",
                            }

                    return {"type": "valid", "file": label_file}

                except Exception as e:
                    return {
                        "type": "invalid",
                        "file": label_file,
                        "reason": f"读取文件失败: {str(e)}",
                    }

            # 批量验证标签文件
            validation_results = process_with_progress(
                label_files, validate_label_file, "验证标签文件"
            )

            for val_result in validation_results:
                if val_result:
                    if val_result["type"] == "invalid":
                        invalid_labels.append(val_result)
                    elif val_result["type"] == "empty":
                        empty_labels.append(val_result)

            stats["invalid_labels"] = len(invalid_labels)
            stats["empty_labels"] = len(empty_labels)

            if invalid_labels:
                issues.append(
                    {
                        "type": "invalid_labels",
                        "count": len(invalid_labels),
                        "examples": invalid_labels,  # 包含所有无效标签用于清理
                        "preview": invalid_labels[:5],  # 只显示前5个例子用于预览
                    }
                )

            if empty_labels:
                issues.append(
                    {
                        "type": "empty_labels",
                        "count": len(empty_labels),
                        "files": [
                            str(item["file"]) for item in empty_labels
                        ],  # 包含所有文件用于清理
                        "preview": [
                            str(item["file"]) for item in empty_labels[:10]
                        ],  # 只显示前10个用于预览
                    }
                )

        # 判断整体验证结果
        self.logger.info(f"验证结束时issues数量: {len(issues)}")
        for i, issue in enumerate(issues):
            self.logger.info(f"Issue {i}: type={issue['type']}, count={issue['count']}")

        if issues:
            result["success"] = False
            self.logger.info("由于存在issues，设置success=False")
        else:
            self.logger.info("没有issues，保持success=True")

        self.logger.info(f"YOLO数据集验证完成: {stats}, success={result['success']}")
        return result

    def _detect_dataset_root(self, input_path: Path) -> Path:
        """智能检测数据集根目录

        如果用户输入的是images或labels子目录，自动向上查找数据集根目录

        Args:
            input_path: 用户输入的路径

        Returns:
            Path: 数据集根目录路径
        """
        current_path = input_path

        # 检查当前路径是否为images或labels子目录
        if current_path.name.lower() in ["images", "labels"]:
            parent_path = current_path.parent

            # 检查父目录是否包含images和labels目录（或至少一个）
            images_dir = parent_path / "images"
            labels_dir = parent_path / "labels"
            classes_file = parent_path / "classes.txt"

            # 如果父目录包含标准YOLO结构，使用父目录作为根目录
            if images_dir.exists() or labels_dir.exists() or classes_file.exists():
                self.logger.info(
                    f"检测到YOLO子目录结构，使用父目录作为数据集根目录: {parent_path}"
                )
                return parent_path

        # 如果当前目录不包含images/labels子目录，但包含图像和标签文件
        # 检查是否需要创建标准目录结构
        images_dir = current_path / "images"
        labels_dir = current_path / "labels"

        if not (images_dir.exists() and labels_dir.exists()):
            # 检查当前目录是否直接包含图像和标签文件
            image_files = get_file_list(
                current_path, self.image_extensions, recursive=False
            )
            txt_files = get_file_list(current_path, [".txt"], recursive=False)
            label_files = [f for f in txt_files if f.name != "classes.txt"]

            if image_files and label_files:
                self.logger.info(
                    f"检测到混合目录结构（图像和标签在同一目录），将使用当前目录: {current_path}"
                )

        return current_path
