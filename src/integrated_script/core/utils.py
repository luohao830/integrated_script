#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py

通用工具函数模块

提供文件操作、路径验证、数据处理等通用功能。
"""

import shutil
from functools import wraps
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

from ..config.exceptions import FileProcessingError, PathError
from .logging_config import get_logger

logger = get_logger(__name__)


def safe_file_operation(operation: str):
    """安全文件操作装饰器

    为文件操作提供统一的错误处理和日志记录。

    Args:
        operation: 操作类型描述
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                logger.debug(f"开始执行文件操作: {operation}")
                result = func(*args, **kwargs)
                logger.debug(f"文件操作完成: {operation}")
                return result
            except PermissionError as e:
                error_msg = f"权限错误 - {operation}: {str(e)}"
                logger.error(error_msg)
                raise FileProcessingError(
                    error_msg,
                    operation=operation,
                    file_path=str(args[0]) if args else "unknown",
                )
            except FileNotFoundError as e:
                error_msg = f"文件未找到 - {operation}: {str(e)}"
                logger.error(error_msg)
                raise FileProcessingError(
                    error_msg,
                    operation=operation,
                    file_path=str(args[0]) if args else "unknown",
                )
            except OSError as e:
                error_msg = f"系统错误 - {operation}: {str(e)}"
                logger.error(error_msg)
                raise FileProcessingError(
                    error_msg,
                    operation=operation,
                    file_path=str(args[0]) if args else "unknown",
                )
            except Exception as e:
                error_msg = f"未知错误 - {operation}: {str(e)}"
                logger.error(error_msg)
                raise FileProcessingError(
                    error_msg,
                    operation=operation,
                    file_path=str(args[0]) if args else "unknown",
                )

        return wrapper

    return decorator


def validate_path(
    path: Union[str, Path],
    must_exist: bool = True,
    must_be_dir: bool = False,
    must_be_file: bool = False,
    create_if_missing: bool = False,
) -> Path:
    """验证路径

    Args:
        path: 路径
        must_exist: 是否必须存在
        must_be_dir: 是否必须是目录
        must_be_file: 是否必须是文件
        create_if_missing: 如果不存在是否创建（仅对目录有效）

    Returns:
        Path: 验证后的路径对象

    Raises:
        PathError: 路径验证失败
    """
    if not path:
        raise PathError("路径不能为空")

    path_obj = Path(path).resolve()

    # 检查存在性
    if must_exist and not path_obj.exists():
        if create_if_missing and must_be_dir:
            try:
                path_obj.mkdir(parents=True, exist_ok=True)
                logger.info(f"创建目录: {path_obj}")
            except Exception as e:
                raise PathError(f"创建目录失败: {str(e)}", path=str(path_obj))
        else:
            raise PathError(f"路径不存在: {path_obj}", path=str(path_obj))

    # 检查类型
    if path_obj.exists():
        if must_be_dir and not path_obj.is_dir():
            raise PathError(f"路径不是目录: {path_obj}", path=str(path_obj))

        if must_be_file and not path_obj.is_file():
            raise PathError(f"路径不是文件: {path_obj}", path=str(path_obj))

    return path_obj


def get_file_list(
    directory: Union[str, Path],
    extensions: Optional[List[str]] = None,
    recursive: bool = False,
    include_hidden: bool = False,
) -> List[Path]:
    """获取文件列表

    Args:
        directory: 目录路径
        extensions: 文件扩展名列表（如 ['.txt', '.jpg']）
        recursive: 是否递归搜索
        include_hidden: 是否包含隐藏文件

    Returns:
        List[Path]: 文件路径列表
    """
    dir_path = validate_path(directory, must_exist=True, must_be_dir=True)

    pattern = "**/*" if recursive else "*"
    files = []

    try:
        for file_path in dir_path.glob(pattern):
            if file_path.is_file():
                # 检查隐藏文件
                if not include_hidden and file_path.name.startswith("."):
                    continue

                # 检查扩展名
                if extensions is None:
                    files.append(file_path)
                else:
                    file_ext = file_path.suffix.lower()
                    if any(file_ext == ext.lower() for ext in extensions):
                        files.append(file_path)

        logger.debug(f"在 {dir_path} 中找到 {len(files)} 个文件")
        return sorted(files)

    except Exception as e:
        raise FileProcessingError(
            f"获取文件列表失败: {str(e)}",
            file_path=str(dir_path),
            operation="list_files",
        )


def cv2_imread_unicode(
    image_path: Union[str, Path], flags: Optional[int] = None
) -> Any:
    """使用OpenCV读取图像（兼容包含中文的路径）"""
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise

    path_obj = Path(image_path)
    try:
        raw = path_obj.read_bytes()
    except Exception as exc:
        logger.warning("读取图像字节失败: %s", exc)
        return None

    array = np.frombuffer(raw, dtype=np.uint8)
    if flags is None:
        flags = cv2.IMREAD_COLOR
    image = cv2.imdecode(array, flags)
    return image


def cv2_imwrite_unicode(
    image_path: Union[str, Path], image: Any, params: Optional[List[int]] = None
) -> bool:
    """使用OpenCV写入图像（兼容包含中文的路径）"""
    try:
        import cv2
    except ImportError:
        raise

    path_obj = Path(image_path)
    suffix = path_obj.suffix or ".png"
    if not suffix.startswith("."):
        suffix = f".{suffix}"

    encode_params = params or []
    success, encoded = cv2.imencode(suffix, image, encode_params)
    if not success:
        return False

    try:
        path_obj.write_bytes(encoded.tobytes())
        return True
    except Exception as exc:
        logger.warning("写入图像字节失败: %s", exc)
        return False


@safe_file_operation("创建目录")
def create_directory(
    path: Union[str, Path], parents: bool = True, exist_ok: bool = True
) -> Path:
    """创建目录

    Args:
        path: 目录路径
        parents: 是否创建父目录
        exist_ok: 如果目录已存在是否忽略

    Returns:
        Path: 创建的目录路径
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=parents, exist_ok=exist_ok)
    # logger.info(f"创建目录:3 {dir_path}")
    return dir_path


@safe_file_operation("复制文件")
def copy_file_safe(
    src: Union[str, Path], dst: Union[str, Path], create_dirs: bool = True
) -> Path:
    """安全复制文件

    Args:
        src: 源文件路径
        dst: 目标文件路径
        create_dirs: 是否创建目标目录

    Returns:
        Path: 目标文件路径
    """
    src_path = validate_path(src, must_exist=True, must_be_file=True)
    dst_path = Path(dst)

    if create_dirs:
        dst_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src_path, dst_path)
    logger.debug(f"复制文件: {src_path} -> {dst_path}")
    return dst_path


@safe_file_operation("移动文件")
def move_file_safe(
    src: Union[str, Path], dst: Union[str, Path], create_dirs: bool = True
) -> Path:
    """安全移动文件

    Args:
        src: 源文件路径
        dst: 目标文件路径
        create_dirs: 是否创建目标目录

    Returns:
        Path: 目标文件路径
    """
    src_path = validate_path(src, must_exist=True, must_be_file=True)
    dst_path = Path(dst)

    if create_dirs:
        dst_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.move(str(src_path), str(dst_path))
    logger.debug(f"移动文件: {src_path} -> {dst_path}")
    return dst_path


@safe_file_operation("删除文件")
def delete_file_safe(path: Union[str, Path], missing_ok: bool = True) -> bool:
    """安全删除文件

    Args:
        path: 文件路径
        missing_ok: 如果文件不存在是否忽略

    Returns:
        bool: 是否成功删除
    """
    file_path = Path(path)

    if not file_path.exists():
        if missing_ok:
            logger.debug(f"文件不存在，跳过删除: {file_path}")
            return False
        else:
            raise FileNotFoundError(f"文件不存在: {file_path}")

    if file_path.is_file():
        file_path.unlink()
        logger.debug(f"删除文件: {file_path}")
        return True
    elif file_path.is_dir():
        shutil.rmtree(file_path)
        logger.debug(f"删除目录: {file_path}")
        return True
    else:
        logger.warning(f"未知文件类型，跳过删除: {file_path}")
        return False


def format_file_size(size_bytes: int) -> str:
    """格式化文件大小

    Args:
        size_bytes: 文件大小（字节）

    Returns:
        str: 格式化的文件大小
    """
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    i = 0
    while size >= 1024 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1

    return f"{size:.1f} {size_names[i]}"


def get_unique_filename(directory: Union[str, Path], filename: str) -> Path:
    """获取唯一的文件名

    如果文件已存在，会在文件名后添加数字后缀。

    Args:
        directory: 目录路径
        filename: 文件名

    Returns:
        Path: 唯一的文件路径
    """
    dir_path = Path(directory)
    file_path = dir_path / filename

    if not file_path.exists():
        return file_path

    # 分离文件名和扩展名
    stem = file_path.stem
    suffix = file_path.suffix

    counter = 1
    while True:
        new_filename = f"{stem}_{counter}{suffix}"
        new_path = dir_path / new_filename
        if not new_path.exists():
            return new_path
        counter += 1
