from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_PRE_PUSH_HOOK_PATH = REPO_ROOT / ".githooks" / "pre-push"
MAKEFILE_PATH = REPO_ROOT / "Makefile"


@pytest.mark.unit
def test_repo_pre_push_hook_runs_local_quality_gate() -> None:
    content = REPO_PRE_PUSH_HOOK_PATH.read_text(encoding="utf-8")

    required_snippets = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'repo_root="$(git rev-parse --show-toplevel)"',
        'cd "$repo_root"',
        "make check-all",
    ]

    for snippet in required_snippets:
        assert snippet in content


@pytest.mark.unit
def test_setup_dev_installs_repo_pre_push_hook() -> None:
    content = MAKEFILE_PATH.read_text(encoding="utf-8")

    required_snippets = [
        "setup-dev: install-dev install-hooks",
        "install-hooks:",
        "git rev-parse --git-path hooks/pre-push",
        "cp .githooks/pre-push",
        "chmod +x",
    ]

    for snippet in required_snippets:
        assert snippet in content
