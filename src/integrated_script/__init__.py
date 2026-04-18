#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated Script Package

一个模块化的数据处理整合脚本项目，提供多种YOLO数据集处理功能。

Modules:
    config: 配置管理和异常定义
    core: 核心基础类和工具函数
    processors: 各种数据处理器
    ui: 用户界面和交互

"""

from .version import get_version

__version__ = get_version()
__author__ = "Integrated Script Team"
__email__ = "team@example.com"
__license__ = "MIT"

from .config.exceptions import (
    ConfigurationError,
    FileProcessingError,
    PathError,
    ProcessingError,
)
from .config.settings import ConfigManager
from .contracts import NormalizedError, OperationResult, normalize_exception

# 导入主要类和函数
from .core.base import BaseProcessor
from .workflows import FileWorkflow, ImageWorkflow, LabelWorkflow, YoloWorkflow

# 定义公共API
__all__ = [
    "BaseProcessor",
    "ProcessingError",
    "PathError",
    "FileProcessingError",
    "ConfigurationError",
    "ConfigManager",
    "OperationResult",
    "NormalizedError",
    "normalize_exception",
    "YoloWorkflow",
    "ImageWorkflow",
    "FileWorkflow",
    "LabelWorkflow",
    "__version__",
]

# 包级别的配置
