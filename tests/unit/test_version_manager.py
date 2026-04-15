from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_PATH = REPO_ROOT / "scripts"
if str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))

from version_manager import VersionManager  # noqa: E402


def test_update_main_version_updates_double_quote_version_pattern(tmp_path: Path) -> None:
    project_root = tmp_path
    (project_root / "src" / "integrated_script").mkdir(parents=True)

    main_file = project_root / "src" / "integrated_script" / "main.py"
    main_file.write_text(
        "parser.add_argument(\"--version\", action=\"version\", version=\"%(prog)s 2.0.0\")\n",
        encoding="utf-8",
    )

    vm = VersionManager(project_root=project_root)
    vm.update_main_version("2.1.0")

    updated = main_file.read_text(encoding="utf-8")
    assert "version=\"%(prog)s 2.1.0\"" in updated


def test_update_pyproject_version_only_updates_project_section(tmp_path: Path) -> None:
    project_root = tmp_path
    pyproject_file = project_root / "pyproject.toml"
    pyproject_file.write_text(
        """
[project]
name = "demo"
version = "1.0.0"

[tool.demo]
version = "keep-me"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    (project_root / "src" / "integrated_script").mkdir(parents=True)
    (project_root / "src" / "integrated_script" / "main.py").write_text("", encoding="utf-8")

    vm = VersionManager(project_root=project_root)
    vm.update_pyproject_version("1.1.0")

    updated = pyproject_file.read_text(encoding="utf-8")
    assert 'version = "1.1.0"' in updated
    assert 'version = "keep-me"' in updated
