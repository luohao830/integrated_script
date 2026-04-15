from pathlib import Path

import pytest

from integrated_script.main import main as package_main


def test_package_main_version_flag_exits_zero(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        package_main(["--version"])

    captured = capsys.readouterr()
    assert exc.value.code == 0
    assert "integrated_script" in captured.out


def test_root_main_has_src_path_insertion_logic() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    content = (repo_root / "main.py").read_text(encoding="utf-8")

    assert "if os.path.isdir(src_path) and src_path not in sys.path:" in content
    assert "sys.path.insert(0, src_path)" in content
    assert "from integrated_script.main import main" in content
