from typing import Any, Dict


def render_result(result: Dict[str, Any]) -> None:
    """渲染通用操作结果。"""
    # 字段名中英文映射
    field_translations = {
        "total_images": "总图像数",
        "total_labels": "总标签数",
        "matched_pairs": "匹配对数",
        "orphaned_images": "孤立图像",
        "orphaned_labels": "孤立标签",
        "invalid_labels": "无效标签",
        "empty_labels": "空标签",
        "dataset_path": "数据集路径",
        "is_valid": "数据集有效性",
        "has_classes_file": "包含类别文件",
        "num_classes": "类别数量",
        "class_names": "类别名称",
        "total_processed": "总处理文件数",
        "invalid_removed": "无效文件数",
        "out_of_bounds_labels": "标签越界数",
        "final_count": "有效文件数",
        "input_path": "输入路径",
        "output_path": "输出路径",
        "project_name": "项目名称",
        "valid": "数据集有效",
        "classes_file": "类别文件路径",
        # 图像处理相关字段
        "total_files": "总文件数",
        "converted_count": "转换成功数",
        "failed_count": "重命名失败数",
        "target_class_only_labels": "仅包含目标类别的标签数",
        "removed_images": "删除的图像数",
        "removed_labels": "删除的标签数",
        "dataset_dir": "数据集目录",
        "images_dir": "图像目录",
        "labels_dir": "标签目录",
        "target_class": "目标类别",
        "total_input_size": "输入总大小",
        "total_output_size": "输出总大小",
        "total_input_size_formatted": "输入总大小",
        "total_output_size_formatted": "输出总大小",
        "overall_compression_ratio": "总体压缩比",
        "input_dir": "输入目录",
        "output_dir": "输出目录",
        "target_format": "目标格式",
        "quality": "图像质量",
        "resized_count": "调整成功数",
        "target_size": "目标尺寸",
        "maintain_aspect_ratio": "保持宽高比",
        "copied_count": "复制成功数",
        "moved_count": "移动成功数",
        "deleted_count": "删除成功数",
        "renamed_count": "重命名成功数",
        # 重命名功能相关字段
        "total_pairs": "总文件对数",
        "rename_pattern": "重命名模式",
        "shuffle_order": "打乱顺序",
        "preview_only": "仅预览",
        "target_dir": "目标目录",
        "prefix": "文件前缀",
        "digits": "数字位数",
        # 图像信息相关字段
        "file_path": "文件路径",
        "file_size": "文件大小(字节)",
        "file_size_formatted": "文件大小",
        "format": "图像格式",
        "width": "宽度",
        "height": "高度",
        "aspect_ratio": "宽高比",
        "total_pixels": "总像素数",
        "mode": "颜色模式",
        "has_transparency": "包含透明度",
    }

    print("\n" + "=" * 50)

    # 检查是否为统计信息结果
    is_statistics_result = (
        "statistics" in result
        and isinstance(result["statistics"], dict)
        and "dataset_path" in result["statistics"]
        and "is_valid" in result["statistics"]
    )

    if is_statistics_result:
        # 这是统计信息结果
        if result["statistics"].get("is_valid", False):
            print("✓ 数据集验证通过")
        else:
            print("⚠ 数据集存在问题")
    elif result.get("success", False):
        print("✓ 操作成功完成")
    else:
        # 对于统计信息结果，不显示操作失败
        if not is_statistics_result:
            print("✗ 操作失败")
            error_message = result.get("error") or result.get("message")
            if error_message:
                print(f"错误信息: {error_message}")

    # 显示统计信息
    if "statistics" in result:
        print("\n统计信息:")
        stats = result["statistics"]
        for key, value in stats.items():
            chinese_key = field_translations.get(key, key)
            # 特殊处理数据集路径，如果路径被调整则显示提示
            if key == "dataset_path":
                print(f"  {chinese_key}: {value}")
                # 检查是否路径被调整（通过比较original_path和dataset_path）
                original_path = stats.get("original_path")
                if original_path and str(original_path) != str(value):
                    print("    💡 已自动调整为数据集根目录")
            elif key != "original_path":  # 不显示original_path字段
                print(f"  {chinese_key}: {value}")

    # 显示其他重要信息
    for key, value in result.items():
        if key not in ["success", "statistics", "message"] and not key.endswith(
            "_list"
        ):
            if isinstance(value, (str, int, float, bool)):
                chinese_key = field_translations.get(key, key)
                # 对布尔值进行中文化
                if isinstance(value, bool):
                    value_text = "是" if value else "否"
                else:
                    value_text = str(value)
                print(f"{chinese_key}: {value_text}")

    # 显示失败文件详情
    if "failed_pairs" in result and result["failed_pairs"]:
        print("\n失败文件详情:")
        for i, failed_item in enumerate(result["failed_pairs"], 1):
            print(f"  {i}. 图像文件: {failed_item.get('img_file', 'N/A')}")
            print(f"     标签文件: {failed_item.get('label_file', 'N/A')}")
            print(f"     失败原因: {failed_item.get('error', 'N/A')}")
            print(f"     失败阶段: {failed_item.get('action', 'N/A')}")
            print()

    print("=" * 50)
