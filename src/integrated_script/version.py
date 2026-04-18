#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
version.py

统一版本读取入口。
优先从已安装包元数据读取，回退到仓库 pyproject.toml。
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

try:
    from importlib.metadata import PackageNotFoundError, version as package_version
except ImportError:  # pragma: no cover
    try:
        from importlib_metadata import (  # type: ignore[import-not-found]
            PackageNotFoundError,
            version as package_version,
        )
    except ImportError:  # pragma: no cover
        PackageNotFoundError = Exception  # type: ignore[assignment]
        package_version = None


_DEFAULT_VERSION = "0.0.0"
_PACKAGE_NAME = "integrated-script"


def _find_pyproject(start_file: Path) -> Optional[Path]:
    """从给定文件路径向上查找 pyproject.toml。"""
    current = start_file.resolve().parent
    for directory in [current, *current.parents]:
        candidate = directory / "pyproject.toml"
        if candidate.exists():
            return candidate
    return None


@lru_cache(maxsize=1)
def get_version() -> str:
    """获取应用版本号。"""
    if package_version is not None:
        try:
            return package_version(_PACKAGE_NAME)
        except PackageNotFoundError:
            pass

    pyproject_file = _find_pyproject(Path(__file__))
    if pyproject_file is None:
        return _DEFAULT_VERSION

    content = pyproject_file.read_text(encoding="utf-8")
    in_project_section = False

    for line in content.splitlines():
        stripped = line.strip()

        if stripped == "[project]":
            in_project_section = True
            continue

        if in_project_section and stripped.startswith("[") and stripped != "[project]":
            break

        if in_project_section:
            match = re.match(r"^\s*version\s*=\s*['\"]([^'\"]+)['\"]", line)
            if match:
                return match.group(1)

    return _DEFAULT_VERSION
