#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
version_manager.py

版本管理脚本

用于自动更新项目版本号并创建Git标签
"""

import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple


class VersionManager:
    """版本管理器"""

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.pyproject_file = self.project_root / "pyproject.toml"
        self.main_file = self.project_root / "src" / "integrated_script" / "main.py"

    def get_current_version(self) -> str:
        """获取当前版本号"""
        if not self.pyproject_file.exists():
            raise FileNotFoundError(
                f"找不到 pyproject.toml 文件: {self.pyproject_file}"
            )

        content = self.pyproject_file.read_text(encoding="utf-8")
        match = re.search(
            r'version\s*=\s*"([^"]*)"|version\s*=\s*\'([^\']*)\'', content
        )

        if not match:
            raise ValueError("无法在 pyproject.toml 中找到版本号")

        # 返回匹配到的组（双引号或单引号）
        return match.group(1) if match.group(1) else match.group(2)

    def parse_version(self, version: str) -> Tuple[int, int, int]:
        """解析版本号"""
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version)
        if not match:
            raise ValueError(f"无效的版本号格式: {version}")

        return tuple(map(int, match.groups()))

    def increment_version(self, version_type: str = "patch") -> str:
        """递增版本号

        Args:
            version_type: 版本类型 (major, minor, patch)

        Returns:
            新的版本号
        """
        current = self.get_current_version()
        major, minor, patch = self.parse_version(current)

        if version_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif version_type == "minor":
            minor += 1
            patch = 0
        elif version_type == "patch":
            patch += 1
        else:
            raise ValueError(f"无效的版本类型: {version_type}")

        return f"{major}.{minor}.{patch}"

    def update_pyproject_version(self, new_version: str) -> None:
        """更新 pyproject.toml 中的版本号"""
        content = self.pyproject_file.read_text(encoding="utf-8")

        # 只更新 [project] 部分的版本号，避免误改工具配置
        # 使用更精确的正则表达式，确保在 [project] 部分内
        lines = content.split("\n")
        new_lines = []
        in_project_section = False

        for line in lines:
            # 检测是否进入 [project] 部分
            if line.strip() == "[project]":
                in_project_section = True
                new_lines.append(line)
            # 检测是否离开 [project] 部分（遇到新的 [section]）
            elif line.strip().startswith("[") and line.strip() != "[project]":
                in_project_section = False
                new_lines.append(line)
            # 在 [project] 部分内且匹配 version = "..." 的行
            elif in_project_section and re.match(r'^\s*version\s*=\s*["\']', line):
                # 替换版本号
                new_line = re.sub(
                    r'(^\s*version\s*=\s*)["\'][^"\'\']*["\']',
                    f'\\1"{new_version}"',
                    line,
                )
                new_lines.append(new_line)
            else:
                new_lines.append(line)

        new_content = "\n".join(new_lines)
        self.pyproject_file.write_text(new_content, encoding="utf-8")
        print(f"✅ 已更新 pyproject.toml 版本号: {new_version}")

    def update_main_version(self, new_version: str) -> None:
        """更新 main.py 中的版本号"""
        if not self.main_file.exists():
            print(f"⚠️  警告: 找不到 main.py 文件: {self.main_file}")
            return

        content = self.main_file.read_text(encoding="utf-8")

        # 新版本入口使用 get_version() 动态读取，无需回写硬编码版本
        if re.search(
            r"version\s*=\s*f?(['\"])%\(prog\)s\s+\{get_version\(\)\}\1",
            content,
        ):
            print("ℹ️  main.py 使用 get_version() 动态读取版本，无需更新")
            return

        # 更新 --version 参数的版本号（兼容单双引号）
        new_content = re.sub(
            r"version\s*=\s*(['\"])%\(prog\)s\s+[^'\"]*\1",
            lambda m: f"version={m.group(1)}%(prog)s {new_version}{m.group(1)}",
            content,
        )

        self.main_file.write_text(new_content, encoding="utf-8")
        print(f"✅ 已更新 main.py 版本号: {new_version}")

    def create_git_tag(self, version: str, message: Optional[str] = None) -> bool:
        """创建 Git 标签"""
        tag_name = f"v{version}"
        tag_message = message or f"Release version {version}"

        try:
            # 检查是否在 Git 仓库中
            subprocess.run(
                ["git", "status"],
                check=True,
                capture_output=True,
                cwd=self.project_root,
            )

            # 检查标签是否已存在
            result = subprocess.run(
                ["git", "tag", "-l", tag_name],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            if result.stdout.strip():
                print(f"⚠️  标签 {tag_name} 已存在")
                return False

            # 创建标签
            subprocess.run(
                ["git", "tag", "-a", tag_name, "-m", tag_message],
                check=True,
                cwd=self.project_root,
            )

            print(f"✅ 已创建 Git 标签: {tag_name}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Git 操作失败: {e}")
            return False
        except FileNotFoundError:
            print("❌ 找不到 Git 命令")
            return False

    def commit_version_changes(self, version: str) -> bool:
        """提交版本更改"""
        try:
            # 添加更改的文件
            subprocess.run(
                ["git", "add", "pyproject.toml"],
                check=True,
                cwd=self.project_root,
            )

            if self.main_file.exists():
                main_file_rel = self.main_file.relative_to(self.project_root).as_posix()
                subprocess.run(
                    ["git", "add", main_file_rel],
                    check=True,
                    cwd=self.project_root,
                )

            # 提交更改
            commit_message = f"chore: bump version to {version}"
            subprocess.run(
                ["git", "commit", "-m", commit_message],
                check=True,
                cwd=self.project_root,
            )

            print(f"✅ 已提交版本更改: {commit_message}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Git 提交失败: {e}")
            return False

    def release(
        self, version_type: str = "patch", message: Optional[str] = None
    ) -> str:
        """执行完整的发布流程

        Args:
            version_type: 版本类型 (major, minor, patch)
            message: 发布消息

        Returns:
            新的版本号
        """
        print(f"🚀 开始发布流程 ({version_type})...")

        # 获取新版本号
        new_version = self.increment_version(version_type)
        print(f"📦 新版本号: {new_version}")

        # 更新版本号
        self.update_pyproject_version(new_version)
        self.update_main_version(new_version)

        # 提交更改
        if not self.commit_version_changes(new_version):
            raise RuntimeError("发布失败: 无法提交版本更改")

        # 创建标签
        if not self.create_git_tag(new_version, message):
            raise RuntimeError("发布失败: 无法创建 Git 标签")

        print(f"🎉 发布完成! 版本: {new_version}")
        print("\n📋 下一步操作:")
        print("   git push origin master")
        print(f"   git push origin v{new_version}")
        print("\n🤖 推送标签后，GitHub Actions 将自动构建和发布 EXE 文件")

        return new_version


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="版本管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python version_manager.py current              # 显示当前版本
  python version_manager.py release patch       # 发布补丁版本
  python version_manager.py release minor       # 发布次要版本
  python version_manager.py release major       # 发布主要版本
  python version_manager.py tag v1.0.0          # 创建指定标签
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # current 命令
    subparsers.add_parser("current", help="显示当前版本号")

    # release 命令
    release_parser = subparsers.add_parser("release", help="发布新版本")
    release_parser.add_argument(
        "type",
        choices=["major", "minor", "patch"],
        default="patch",
        nargs="?",
        help="版本类型 (默认: patch)",
    )
    release_parser.add_argument("-m", "--message", help="发布消息")

    # tag 命令
    tag_parser = subparsers.add_parser("tag", help="创建 Git 标签")
    tag_parser.add_argument("version", help="版本号 (如: v1.0.0)")
    tag_parser.add_argument("-m", "--message", help="标签消息")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    vm = VersionManager()

    try:
        if args.command == "current":
            version = vm.get_current_version()
            print(f"当前版本: {version}")

        elif args.command == "release":
            vm.release(args.type, args.message)

        elif args.command == "tag":
            version = args.version.lstrip("v")
            if vm.create_git_tag(version, args.message):
                print(f"✅ 标签创建成功: v{version}")
            else:
                sys.exit(1)

    except Exception as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
