#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""image_processor.py

图像处理器 façade。
"""

from ..config.exceptions import ProcessingError
from .image import core as image_core
from .image.core import _process_batch_worker

CV2_AVAILABLE = image_core.CV2_AVAILABLE
PIL_AVAILABLE = image_core.PIL_AVAILABLE
TQDM_AVAILABLE = image_core.TQDM_AVAILABLE


class ImageProcessor(image_core.ImageProcessor):
    """兼容 façade：保持旧模块级开关可被 monkeypatch。"""

    def _check_dependencies(self) -> None:  # type: ignore[override]
        if not CV2_AVAILABLE and not PIL_AVAILABLE:
            raise ProcessingError(
                "图像处理需要安装 opencv-python 或 Pillow 库\n"
                "请运行: pip install opencv-python Pillow",
                error_code="MISSING_DEPENDENCIES",
                context={"missing_dependencies": ["opencv-python", "Pillow"]},
            )

        logger = getattr(self, "logger", None)
        if logger is not None:
            if not PIL_AVAILABLE:
                logger.warning("Pillow 库未安装，某些功能可能不可用")
            if not CV2_AVAILABLE:
                logger.warning("OpenCV 库未安装，某些功能可能不可用")


__all__ = [
    "CV2_AVAILABLE",
    "PIL_AVAILABLE",
    "TQDM_AVAILABLE",
    "ImageProcessor",
    "_process_batch_worker",
]
