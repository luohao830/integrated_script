#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ui模块

用户界面相关功能

提供交互式界面等用户交互功能。
"""

from ..version import get_version
from .interactive import InteractiveInterface
from .menu import MenuSystem

__all__ = ["InteractiveInterface", "MenuSystem"]

__version__ = get_version()
__author__ = "Integrated Script Team"
