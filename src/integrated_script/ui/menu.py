#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
menu.py

菜单系统

提供菜单显示和导航功能。
"""

import os
from typing import Any, Dict, Optional

from ..config.exceptions import UserInterruptError


class MenuSystem:
    """菜单系统

    提供菜单显示和导航功能。

    Attributes:
        main_menu (Dict): 主菜单配置
        current_menu (Dict): 当前菜单
        menu_stack (List): 菜单栈，用于返回上级菜单
    """

    def __init__(self):
        """初始化菜单系统"""
        self.main_menu = None
        self.current_menu = None
        self.menu_stack = []

    def set_main_menu(self, menu: Dict[str, Any]) -> None:
        """设置主菜单

        Args:
            menu: 菜单配置字典
                {
                    'title': '菜单标题',
                    'options': [
                        ('选项1', callback1),
                        ('选项2', callback2),
                        ...
                    ]
                }
        """
        self.main_menu = menu
        self.current_menu = menu

    def show_menu(self, menu: Optional[Dict[str, Any]] = None) -> None:
        """显示菜单

        Args:
            menu: 要显示的菜单，如果为None则显示当前菜单
        """
        if menu:
            self.menu_stack.append(self.current_menu)
            self.current_menu = menu

        while True:
            try:
                self._display_current_menu()
                choice = self._get_user_choice()

                if choice == 0:  # 返回上级菜单或退出
                    if self.menu_stack:
                        self.current_menu = self.menu_stack.pop()
                        continue
                    else:
                        # 在主菜单选择0时直接退出程序
                        import sys

                        sys.exit(0)

                # 执行选择的操作
                options = self.current_menu.get("options", [])
                if 1 <= choice <= len(options):
                    option_name, callback = options[choice - 1]

                    if callback is None:  # 返回上级菜单
                        if self.menu_stack:
                            self.current_menu = self.menu_stack.pop()
                        else:
                            break
                    elif callable(callback):
                        try:
                            callback()
                        except UserInterruptError:
                            # 用户中断时直接返回，不显示额外信息
                            pass
                        except Exception as e:
                            print(f"\n操作执行失败: {e}")
                            self._pause()
                    else:
                        print(f"\n无效的回调函数: {callback}")
                        self._pause()
                else:
                    print(f"\n无效选择: {choice}")
                    self._pause()

            except UserInterruptError:
                # 用户中断时直接返回上级菜单，不显示额外信息
                if self.menu_stack:
                    self.current_menu = self.menu_stack.pop()
                else:
                    # 在主菜单时直接退出程序
                    import sys

                    sys.exit(0)
            except Exception as e:
                print(f"\n\n菜单系统错误: {e}")
                self._pause()

    def _display_current_menu(self) -> None:
        """显示当前菜单"""
        self._clear_screen()

        # 显示标题
        title = self.current_menu.get("title", "菜单")
        print("\n" + "=" * 60)
        print(f"{title:^60}")
        print("=" * 60)

        # 显示选项
        options = self.current_menu.get("options", [])
        for i, (option_name, _) in enumerate(options, 1):
            print(f"{i:2d}. {option_name}")

        # 显示返回选项
        if self.menu_stack:
            print(" 0. 返回上级菜单")
        else:
            print(" 0. 退出程序")

        print("=" * 60)

    def _get_user_choice(self) -> int:
        """获取用户选择

        Returns:
            int: 用户选择的选项编号
        """
        while True:
            try:
                choice_str = input("\n请选择操作 (输入数字): ").strip()

                if not choice_str:
                    print("请输入有效的数字")
                    continue

                choice = int(choice_str)

                options_count = len(self.current_menu.get("options", []))
                if 0 <= choice <= options_count:
                    return choice
                else:
                    print(f"请输入 0 到 {options_count} 之间的数字")

            except ValueError:
                print("请输入有效的数字")
            except KeyboardInterrupt:
                raise UserInterruptError("用户中断操作")
            except EOFError:
                raise UserInterruptError("输入结束")

    def _clear_screen(self) -> None:
        """清屏"""
        try:
            if os.name == "nt":  # Windows
                os.system("cls")
            else:  # Unix/Linux/MacOS
                os.system("clear")
        except Exception:
            # 如果清屏失败，打印空行
            print("\n" * 50)

    def _pause(self) -> None:
        """暂停等待用户按键"""
        try:
            input("\n按回车键继续...")
        except KeyboardInterrupt:
            pass
        except EOFError:
            pass

    def run(self) -> None:
        """运行菜单系统"""
        if not self.main_menu:
            raise ValueError("未设置主菜单")

        try:
            self.show_menu()
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
        except Exception as e:
            print(f"\n\n菜单系统运行错误: {e}")
