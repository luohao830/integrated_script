from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_PATH = REPO_ROOT / "scripts"
if str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))

import release as release_module  # noqa: E402


class _DummyVM:
    def __init__(self, version: str = "2.0.3"):
        self._version = version

    def release(self, _version_type: str, _message):
        return self._version


def _build_manager(tmp_path: Path, version: str = "2.0.3") -> release_module.ReleaseManager:
    manager = release_module.ReleaseManager(project_root=tmp_path)
    manager.vm = _DummyVM(version=version)
    return manager


def test_release_auto_push_returns_false_when_wait_for_actions_fails(tmp_path: Path, monkeypatch) -> None:
    manager = _build_manager(tmp_path)

    monkeypatch.setattr(manager, "check_git_status", lambda: True)
    monkeypatch.setattr(manager, "run_tests", lambda: True)
    monkeypatch.setattr(manager, "build_executable", lambda: True)
    monkeypatch.setattr(manager, "test_executable", lambda: True)
    monkeypatch.setattr(manager, "push_to_github", lambda _version: True)
    monkeypatch.setattr(manager, "wait_for_github_actions", lambda _version: False)

    success = manager.release(auto_push=True)

    assert success is False


def test_release_auto_push_returns_true_when_wait_for_actions_succeeds(tmp_path: Path, monkeypatch) -> None:
    manager = _build_manager(tmp_path)

    monkeypatch.setattr(manager, "check_git_status", lambda: True)
    monkeypatch.setattr(manager, "run_tests", lambda: True)
    monkeypatch.setattr(manager, "build_executable", lambda: True)
    monkeypatch.setattr(manager, "test_executable", lambda: True)
    monkeypatch.setattr(manager, "push_to_github", lambda _version: True)
    monkeypatch.setattr(manager, "wait_for_github_actions", lambda _version: True)

    success = manager.release(auto_push=True)

    assert success is True


def test_wait_for_github_actions_returns_false_when_timeout(tmp_path: Path, monkeypatch) -> None:
    manager = _build_manager(tmp_path)

    tick = {"value": -20}

    def _fake_time() -> int:
        tick["value"] += 20
        return tick["value"]

    monkeypatch.setattr(release_module.time, "time", _fake_time)
    monkeypatch.setattr(release_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        manager,
        "get_github_workflow_status",
        lambda _version: {"status": "not_found"},
    )

    result = manager.wait_for_github_actions("2.0.3", timeout=30)

    assert result is False


def test_wait_for_github_actions_returns_false_when_repeated_old_workflow(tmp_path: Path, monkeypatch) -> None:
    manager = _build_manager(tmp_path)

    tick = {"value": 0}

    def _fake_time() -> int:
        tick["value"] += 1
        return tick["value"]

    monkeypatch.setattr(release_module.time, "time", _fake_time)
    monkeypatch.setattr(release_module.time, "sleep", lambda _seconds: None)

    status = {
        "status": "completed",
        "conclusion": "success",
        "head_branch": "v2.0.2",
    }
    monkeypatch.setattr(manager, "get_github_workflow_status", lambda _version: status)

    result = manager.wait_for_github_actions("2.0.3", timeout=300)

    assert result is False


def test_wait_for_github_actions_returns_false_when_api_error(tmp_path: Path, monkeypatch) -> None:
    manager = _build_manager(tmp_path)

    timeline = iter([0, 1])
    monkeypatch.setattr(release_module.time, "time", lambda: next(timeline))
    monkeypatch.setattr(release_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        manager,
        "get_github_workflow_status",
        lambda _version: {"status": "error", "error": "boom"},
    )

    result = manager.wait_for_github_actions("2.0.3", timeout=300)

    assert result is False
