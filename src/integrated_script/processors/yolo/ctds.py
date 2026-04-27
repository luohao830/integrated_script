"""CTDS flow internals for YOLO processor Phase 3 migration slices."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

from ...config.exceptions import DatasetError
from ...core.progress import progress_context
from ...core.utils import (
    copy_file_safe,
    create_directory,
    move_file_safe,
    validate_path,
)

if TYPE_CHECKING:
    from ..yolo_processor import YOLOProcessor


def get_project_name(
    processor: "YOLOProcessor",
    obj_names_path: Path,
    manual_name: Optional[str] = None,
) -> str:
    """获取项目名称。"""
    if manual_name and manual_name.strip():
        return manual_name.strip()

    try:
        with obj_names_path.open("r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            if lines:
                project_name = "-".join(lines)
                processor.logger.info(f"从 obj.names 获取到项目名称: {project_name}")
                return project_name
    except Exception as e:
        processor.logger.warning(f"读取 obj.names 失败: {str(e)}")

    return "dataset"


def process_ctds_dataset_internal(
    processor: "YOLOProcessor",
    input_path: str,
    output_name: Optional[str] = None,
    keep_empty_labels: bool = False,
) -> Dict[str, Any]:
    """CTDS 数据转 YOLO 格式，先返回预检测阶段结果。"""
    try:
        input_dir = validate_path(input_path, must_exist=True, must_be_dir=True)
        processor.logger.info(f"开始处理CTDS数据集: {input_dir}")

        obj_names_path = input_dir / "obj.names"
        obj_train_data_path = input_dir / "obj_train_data"

        if not obj_names_path.exists():
            raise DatasetError("未找到obj.names文件", dataset_path=str(input_dir))

        if not obj_train_data_path.exists():
            raise DatasetError("未找到obj_train_data目录", dataset_path=str(input_dir))

        project_name = get_project_name(processor, obj_names_path, output_name)

        processor.logger.info("正在预检测数据集类型...")
        pre_detection_result = processor.detect_dataset_type(str(obj_train_data_path))

        if pre_detection_result.get("success"):
            return {
                "success": True,
                "stage": "pre_detection",
                "input_path": str(input_dir),
                "project_name": project_name,
                "pre_detection_result": pre_detection_result,
                "obj_names_path": str(obj_names_path),
                "obj_train_data_path": str(obj_train_data_path),
                "keep_empty_labels": keep_empty_labels,
            }

        processor.logger.warning("预检测失败，将使用默认检测类型进行处理")
        return execute_ctds_processing_internal(
            processor,
            input_dir=input_dir,
            project_name=project_name,
            obj_names_path=obj_names_path,
            obj_train_data_path=obj_train_data_path,
            confirmed_type="detection",
            pre_detection_result=pre_detection_result,
            keep_empty_labels=keep_empty_labels,
        )

    except Exception as e:
        processor.logger.error(f"CTDS数据处理失败: {str(e)}")
        raise DatasetError(
            f"CTDS数据处理失败: {str(e)}",
            dataset_path=str(input_dir) if "input_dir" in locals() else input_path,
        )


def continue_ctds_processing_internal(
    processor: "YOLOProcessor",
    pre_result: Dict[str, Any],
    confirmed_type: str,
    keep_empty_labels: Optional[bool] = None,
) -> Dict[str, Any]:
    """在用户确认数据集类型后继续 CTDS 处理。"""
    try:
        input_dir = Path(pre_result["input_path"])
        project_name = pre_result["project_name"]
        obj_names_path = Path(pre_result["obj_names_path"])
        obj_train_data_path = Path(pre_result["obj_train_data_path"])
        pre_detection_result = pre_result["pre_detection_result"]

        return execute_ctds_processing_internal(
            processor,
            input_dir=input_dir,
            project_name=project_name,
            obj_names_path=obj_names_path,
            obj_train_data_path=obj_train_data_path,
            confirmed_type=confirmed_type,
            pre_detection_result=pre_detection_result,
            keep_empty_labels=(
                pre_result.get("keep_empty_labels", False)
                if keep_empty_labels is None
                else keep_empty_labels
            ),
        )

    except Exception as e:
        raise DatasetError(
            f"CTDS数据处理失败: {str(e)}",
            dataset_path=pre_result.get("input_path", ""),
        )


def execute_ctds_processing_internal(
    processor: "YOLOProcessor",
    input_dir: Path,
    project_name: str,
    obj_names_path: Path,
    obj_train_data_path: Path,
    confirmed_type: str,
    pre_detection_result: Dict[str, Any],
    keep_empty_labels: bool = False,
) -> Dict[str, Any]:
    """执行 CTDS 数据处理核心逻辑。"""
    try:
        output_dir = input_dir.parent / project_name
        create_directory(output_dir)

        labels_dir = output_dir / "labels"
        images_dir = output_dir / "images"
        create_directory(labels_dir)
        create_directory(images_dir)

        result: Dict[str, Any] = {
            "success": True,
            "input_path": str(input_dir),
            "output_path": str(output_dir),
            "project_name": project_name,
            "pre_detection_result": pre_detection_result,
            "confirmed_type": confirmed_type,
            "processed_files": {"images": [], "labels": [], "classes_file": None},
            "invalid_files": [],
            "invalid_details": {
                "empty_labels": [],
                "invalid_labels": [],
                "out_of_bounds_labels": [],
                "missing_images": [],
                "missing_labels": [],
            },
            "statistics": {
                "total_processed": 0,
                "invalid_removed": 0,
                "final_count": 0,
                "empty_removed": 0,
                "out_of_bounds_labels": 0,
                "missing_images": 0,
                "missing_labels": 0,
            },
        }
        processed_files: Dict[str, Any] = cast(
            Dict[str, Any], result["processed_files"]
        )
        invalid_details: Dict[str, Any] = cast(
            Dict[str, Any], result["invalid_details"]
        )
        invalid_files: List[str] = cast(List[str], result["invalid_files"])
        statistics: Dict[str, int] = cast(Dict[str, int], result["statistics"])
        processed_images: List[str] = cast(List[str], processed_files["images"])
        processed_labels: List[str] = cast(List[str], processed_files["labels"])
        empty_labels: List[str] = cast(List[str], invalid_details["empty_labels"])
        invalid_labels: List[str] = cast(List[str], invalid_details["invalid_labels"])
        out_of_bounds_labels: List[str] = cast(
            List[str], invalid_details["out_of_bounds_labels"]
        )
        missing_images: List[str] = cast(List[str], invalid_details["missing_images"])
        missing_labels: List[str] = cast(List[str], invalid_details["missing_labels"])
        hard_error = False

        all_files = list(obj_train_data_path.iterdir())
        txt_files = [
            f
            for f in all_files
            if f.suffix.lower() == ".txt" and f.name.lower() != "train.txt"
        ]
        label_stems = {f.stem for f in txt_files}
        img_files = [
            f
            for f in all_files
            if f.suffix.lower() in {ext.lower() for ext in processor.image_extensions}
        ]
        img_by_stem = {f.stem: f for f in img_files}

        processor.logger.info(f"找到 {len(txt_files)} 个标签文件")
        processor.logger.info(f"找到 {len(img_files)} 个图像文件")
        processor.logger.info(f"使用确认的数据集类型进行验证: {confirmed_type}")

        count = 1
        invalid_count = 0

        with progress_context(len(txt_files), "处理CTDS数据") as progress:
            for txt_file in txt_files:
                try:
                    validation_status = processor._validate_ctds_label_file(
                        txt_file,
                        confirmed_type,
                    )

                    if validation_status == "empty" and not keep_empty_labels:
                        invalid_count += 1
                        statistics["empty_removed"] += 1
                        invalid_files.append(str(txt_file))
                        empty_labels.append(str(txt_file))
                        progress.update_progress(1)
                        continue

                    if validation_status == "out_of_bounds":
                        invalid_count += 1
                        statistics["out_of_bounds_labels"] += 1
                        invalid_files.append(str(txt_file))
                        invalid_labels.append(str(txt_file))
                        out_of_bounds_labels.append(str(txt_file))
                        progress.update_progress(1)
                        continue

                    if validation_status == "invalid":
                        invalid_count += 1
                        invalid_files.append(str(txt_file))
                        invalid_labels.append(str(txt_file))
                        progress.update_progress(1)
                        continue

                    base_name = txt_file.stem
                    img_file = img_by_stem.get(base_name)

                    if img_file:
                        new_txt_name = f"{project_name}-{count:05d}.txt"
                        new_txt_path = labels_dir / new_txt_name
                        move_file_safe(txt_file, new_txt_path)
                        processed_labels.append(str(new_txt_path))

                        new_img_name = f"{project_name}-{count:05d}{img_file.suffix}"
                        new_img_path = images_dir / new_img_name
                        move_file_safe(img_file, new_img_path)
                        processed_images.append(str(new_img_path))
                    else:
                        invalid_count += 1
                        statistics["missing_images"] += 1
                        invalid_files.append(str(txt_file))
                        missing_images.append(str(txt_file))
                        processor.logger.warning(
                            f"未找到标签文件 {txt_file.name} 对应的图像文件，已跳过"
                        )
                        progress.update_progress(1)
                        continue

                    count += 1
                    progress.update_progress(1)

                except Exception as e:
                    processor.logger.error(f"处理文件失败 {txt_file}: {str(e)}")
                    hard_error = True

        total_files = len(txt_files)
        valid_files = count - 1
        statistics["total_processed"] = total_files
        statistics["invalid_removed"] = invalid_count
        statistics["final_count"] = valid_files

        unmatched_images = [f for f in img_files if f.stem not in label_stems]
        statistics["missing_labels"] = len(unmatched_images)
        if unmatched_images:
            invalid_files.extend(str(f) for f in unmatched_images)
            missing_labels.extend(str(f) for f in unmatched_images)

        statistics["invalid_removed"] = invalid_count + statistics["missing_labels"]
        statistics["total_processed"] = (
            statistics["final_count"] + statistics["invalid_removed"]
        )

        processor.logger.info(
            f"CTDS数据处理完成 - 总文件数: {statistics['total_processed']}, 有效文件: {valid_files}, 无效文件: {statistics['invalid_removed']}"
        )
        if statistics["missing_images"] > 0:
            processor.logger.warning(
                f"未找到图像的标签文件数: {statistics['missing_images']}"
            )
        if statistics["missing_labels"] > 0:
            processor.logger.warning(
                f"未找到标签的图像文件数: {statistics['missing_labels']}"
            )

        try:
            import importlib.util

            if importlib.util.find_spec("cv2") is None:
                raise ImportError
            processor._convert_images_with_opencv(images_dir)
        except ImportError:
            processor.logger.warning("OpenCV不可用，跳过图像转换")
        except Exception as e:
            processor.logger.warning(f"图像转换失败: {str(e)}")

        classes_file_path = output_dir / "classes.txt"
        try:
            copy_file_safe(obj_names_path, classes_file_path)
            processed_files["classes_file"] = str(classes_file_path)
            processor.logger.info(
                f"已复制 {obj_names_path.name} 到 {classes_file_path.name}"
            )
        except Exception as e:
            processor.logger.error(f"复制classes文件失败: {str(e)}")
            hard_error = True

        final_count = count - 1
        final_project_name = f"{project_name}-{final_count:05d}"
        final_output_dir = input_dir.parent / final_project_name

        if output_dir != final_output_dir:
            target_dir = final_output_dir
            if target_dir.exists():
                suffix = 1
                while True:
                    candidate = Path(f"{final_output_dir}-{suffix}")
                    if not candidate.exists():
                        target_dir = candidate
                        break
                    suffix += 1
                processor.logger.warning(
                    f"目标目录已存在，使用新名称: {target_dir.name}"
                )

            try:
                output_dir.rename(target_dir)
                result["output_path"] = str(target_dir)
                result["project_name"] = target_dir.name
                processor.logger.info(f"项目文件夹已重命名为: {target_dir.name}")
            except Exception as e:
                processor.logger.error(f"重命名项目文件夹失败: {str(e)}")
                hard_error = True

        statistics["invalid_removed"] = invalid_count + statistics["missing_labels"]
        statistics["final_count"] = final_count
        statistics["total_processed"] = final_count + statistics["invalid_removed"]

        result["detected_dataset_type"] = confirmed_type
        result["detection_confidence"] = pre_detection_result.get("confidence", 1.0)
        result["dataset_type_detection"] = pre_detection_result

        result["success"] = not hard_error
        return result

    except Exception as e:
        raise DatasetError(f"CTDS数据处理失败: {str(e)}", dataset_path=str(input_dir))
