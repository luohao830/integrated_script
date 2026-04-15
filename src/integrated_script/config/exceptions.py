#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exceptions.py

自定义异常类定义模块

定义了项目中使用的各种自定义异常类，用于更精确的错误处理和调试。
"""

from typing import Any, Dict, Optional


class ProcessingError(Exception):
    """处理过程中的基础异常类

    所有处理相关异常的基类。

    Attributes:
        message (str): 错误消息
        error_code (str): 错误代码
        context (dict): 错误上下文信息
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "PROCESSING_ERROR"
        self.context: Dict[str, Any] = context or {}

    def __str__(self):
        if self.context:
            return f"{self.message} (Code: {self.error_code}, Context: {self.context})"
        return f"{self.message} (Code: {self.error_code})"


class PathError(ProcessingError):
    """路径相关错误

    当路径不存在、无权限访问或格式错误时抛出。

    Attributes:
        path (str): 出错的路径
    """

    def __init__(self, message: str, path: Optional[str] = None, **kwargs: Any):
        super().__init__(message, error_code="PATH_ERROR", **kwargs)
        self.path = path
        if path:
            self.context["path"] = path


class FileProcessingError(ProcessingError):
    """文件处理错误

    文件读写、权限、格式等相关错误。

    Attributes:
        file_path (str): 出错的文件路径
        operation (str): 执行的操作
    """

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(message, error_code="FILE_PROCESSING_ERROR", **kwargs)
        self.file_path = file_path
        self.operation = operation
        if file_path:
            self.context["file_path"] = file_path
        if operation:
            self.context["operation"] = operation


class ConfigurationError(ProcessingError):
    """配置错误

    配置文件格式错误、缺少必需配置项等。

    Attributes:
        config_key (str): 出错的配置键
        config_file (str): 配置文件路径
    """

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_file: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(message, error_code="CONFIGURATION_ERROR", **kwargs)
        self.config_key = config_key
        self.config_file = config_file
        if config_key:
            self.context["config_key"] = config_key
        if config_file:
            self.context["config_file"] = config_file


class ValidationError(ProcessingError):
    """数据验证错误

    数据格式、内容验证失败时抛出。

    Attributes:
        validation_type (str): 验证类型
        expected (str): 期望值
        actual (str): 实际值
    """

    def __init__(
        self,
        message: str,
        validation_type: Optional[str] = None,
        expected: Optional[str] = None,
        actual: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(message, error_code="VALIDATION_ERROR", **kwargs)
        self.validation_type = validation_type
        self.expected = expected
        self.actual = actual
        if validation_type:
            self.context["validation_type"] = validation_type
        if expected:
            self.context["expected"] = expected
        if actual:
            self.context["actual"] = actual


class DatasetError(ProcessingError):
    """数据集相关错误

    YOLO数据集结构、格式等相关错误。

    Attributes:
        dataset_path (str): 数据集路径
        dataset_type (str): 数据集类型
    """

    def __init__(
        self,
        message: str,
        dataset_path: Optional[str] = None,
        dataset_type: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(message, error_code="DATASET_ERROR", **kwargs)
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        if dataset_path:
            self.context["dataset_path"] = dataset_path
        if dataset_type:
            self.context["dataset_type"] = dataset_type


class UserInterruptError(ProcessingError):
    """用户中断错误

    用户主动中断操作时抛出。
    """

    def __init__(self, message: str = "操作被用户中断", **kwargs: Any):
        super().__init__(message, error_code="USER_INTERRUPT", **kwargs)


# 异常映射字典，用于根据错误类型快速创建异常
