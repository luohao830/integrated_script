from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, List, Tuple

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


class _RecordingVM(_DummyVM):
    def __init__(self, version: str = "2.0.3"):
        super().__init__(version)
        self.calls: List[Tuple[str, Any]] = []

    def release(self, version_type: str, message):
        self.calls.append((version_type, message))
        return self._version


class _RecordingLocalExecutor:
    def __init__(self, responses: List[Any]):
        self._responses = list(responses)
        self.calls: List[dict] = []

    def run(
        self,
        cmd,
        cwd=None,
        capture_output=False,
        text=False,
        check=False,
        timeout=None,
    ):
        self.calls.append(
            {
                "cmd": cmd,
                "cwd": cwd,
                "capture_output": capture_output,
                "text": text,
                "check": check,
                "timeout": timeout,
            }
        )

        response = self._responses.pop(0) if self._responses else None
        if isinstance(response, Exception):
            raise response
        if isinstance(response, dict):
            result = SimpleNamespace(
                stdout=response.get("stdout", ""),
                stderr=response.get("stderr", ""),
                returncode=response.get("returncode", 0),
            )
            if check and result.returncode != 0:
                raise release_module.subprocess.CalledProcessError(
                    returncode=result.returncode,
                    cmd=cmd,
                    output=result.stdout,
                    stderr=result.stderr,
                )
            return result
        if response is None:
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        return response


class _FakeGitHubClient:
    def __init__(self, payload: dict):
        self.payload = payload
        self.call_count = 0

    def get_workflow_runs(self) -> dict:
        self.call_count += 1
        return self.payload


class _FakeClock:
    def __init__(self, time_points: List[int]):
        self._time_points = iter(time_points)
        self.sleep_calls: List[int] = []

    def now(self) -> int:
        return next(self._time_points)

    def sleep(self, seconds: int) -> None:
        self.sleep_calls.append(seconds)


def _build_manager(
    tmp_path: Path,
    version: str = "2.0.3",
    **kwargs,
) -> release_module.ReleaseManager:
    manager = release_module.ReleaseManager(project_root=tmp_path, **kwargs)
    manager.vm = _DummyVM(version=version)
    return manager


def test_release_auto_push_returns_false_when_wait_for_actions_fails(
    tmp_path: Path, monkeypatch
) -> None:
    manager = _build_manager(tmp_path)

    monkeypatch.setattr(manager, "check_git_status", lambda: True)
    monkeypatch.setattr(manager, "run_tests", lambda: True)
    monkeypatch.setattr(manager, "build_executable", lambda: True)
    monkeypatch.setattr(manager, "test_executable", lambda: True)
    monkeypatch.setattr(manager, "push_to_github", lambda _version: True)
    monkeypatch.setattr(manager, "wait_for_github_actions", lambda _version: False)

    success = manager.release(auto_push=True)

    assert success is False


def test_release_auto_push_returns_true_when_wait_for_actions_succeeds(
    tmp_path: Path, monkeypatch
) -> None:
    manager = _build_manager(tmp_path)

    monkeypatch.setattr(manager, "check_git_status", lambda: True)
    monkeypatch.setattr(manager, "run_tests", lambda: True)
    monkeypatch.setattr(manager, "build_executable", lambda: True)
    monkeypatch.setattr(manager, "test_executable", lambda: True)
    monkeypatch.setattr(manager, "push_to_github", lambda _version: True)
    monkeypatch.setattr(manager, "wait_for_github_actions", lambda _version: True)

    success = manager.release(auto_push=True)

    assert success is True


def test_wait_for_github_actions_returns_false_when_timeout(
    tmp_path: Path, monkeypatch
) -> None:
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


def test_wait_for_github_actions_returns_false_when_workflow_failed(
    tmp_path: Path, monkeypatch
) -> None:
    manager = _build_manager(tmp_path)

    tick = {"value": 0}

    def _fake_time() -> int:
        tick["value"] += 1
        return tick["value"]

    monkeypatch.setattr(release_module.time, "time", _fake_time)
    monkeypatch.setattr(release_module.time, "sleep", lambda _seconds: None)

    status = {
        "status": "completed",
        "conclusion": "failure",
        "head_branch": "v2.0.3",
    }
    monkeypatch.setattr(manager, "get_github_workflow_status", lambda _version: status)

    result = manager.wait_for_github_actions("2.0.3", timeout=300)

    assert result is False


def test_wait_for_github_actions_returns_false_when_api_error(
    tmp_path: Path, monkeypatch
) -> None:
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


def test_push_to_github_uses_current_branch_and_version_tag(tmp_path: Path) -> None:
    executor = _RecordingLocalExecutor(
        [
            {"stdout": "master\n"},
            {},
            {},
        ]
    )
    manager = _build_manager(tmp_path, local_executor=executor)

    success = manager.push_to_github("2.0.3")

    assert success is True
    assert [call["cmd"] for call in executor.calls] == [
        ["git", "branch", "--show-current"],
        ["git", "push", "origin", "master"],
        ["git", "push", "origin", "v2.0.3"],
    ]


def test_get_github_workflow_status_prefers_exact_target_run(tmp_path: Path) -> None:
    github_client = _FakeGitHubClient(
        {
            "workflow_runs": [
                {
                    "status": "completed",
                    "conclusion": "success",
                    "head_branch": "main",
                    "event": "push",
                    "html_url": "latest-url",
                    "created_at": "now",
                    "updated_at": "now",
                    "name": "latest",
                },
                {
                    "status": "completed",
                    "conclusion": "success",
                    "head_branch": "v2.0.3",
                    "event": "push",
                    "html_url": "target-url",
                    "created_at": "then",
                    "updated_at": "then",
                    "name": "target",
                },
            ]
        }
    )
    manager = _build_manager(tmp_path, github_client=github_client)

    status = manager.get_github_workflow_status("2.0.3")

    assert status["head_branch"] == "v2.0.3"
    assert status["html_url"] == "target-url"
    assert github_client.call_count == 1


def test_get_github_workflow_status_returns_not_found_when_target_absent(
    tmp_path: Path,
) -> None:
    github_client = _FakeGitHubClient(
        {
            "workflow_runs": [
                {
                    "status": "completed",
                    "conclusion": "success",
                    "head_branch": "main",
                    "event": "push",
                    "html_url": "latest-url",
                    "created_at": "now",
                    "updated_at": "now",
                    "name": "latest",
                }
            ]
        }
    )
    manager = _build_manager(tmp_path, github_client=github_client)

    status = manager.get_github_workflow_status("2.0.3")

    assert status == {"status": "not_found"}
    assert github_client.call_count == 1


def test_wait_for_github_actions_returns_true_on_target_workflow_success(
    tmp_path: Path, monkeypatch
) -> None:
    clock = _FakeClock([0, 1, 2, 3, 4, 5])
    manager = _build_manager(tmp_path, clock=clock)

    statuses = iter(
        [
            {"status": "not_found"},
            {"status": "in_progress", "head_branch": "v2.0.3"},
            {
                "status": "completed",
                "conclusion": "success",
                "head_branch": "v2.0.3",
                "html_url": "target-url",
            },
        ]
    )
    monkeypatch.setattr(manager, "get_github_workflow_status", lambda _version: next(statuses))

    result = manager.wait_for_github_actions("2.0.3", timeout=30)

    assert result is True
    assert clock.sleep_calls[0] == 15


def test_release_stops_when_git_status_fails(tmp_path: Path, monkeypatch) -> None:
    manager = _build_manager(tmp_path)

    called = {
        "run_tests": False,
        "build_executable": False,
        "test_executable": False,
        "vm_release": False,
    }

    monkeypatch.setattr(manager, "check_git_status", lambda: False)
    monkeypatch.setattr(
        manager,
        "run_tests",
        lambda: called.__setitem__("run_tests", True) or True,
    )
    monkeypatch.setattr(
        manager,
        "build_executable",
        lambda: called.__setitem__("build_executable", True) or True,
    )
    monkeypatch.setattr(
        manager,
        "test_executable",
        lambda: called.__setitem__("test_executable", True) or True,
    )
    monkeypatch.setattr(
        manager.vm,
        "release",
        lambda *_args: called.__setitem__("vm_release", True) or "2.0.4",
    )

    success = manager.release(auto_push=True)

    assert success is False
    assert called == {
        "run_tests": False,
        "build_executable": False,
        "test_executable": False,
        "vm_release": False,
    }


def test_release_auto_push_threads_new_version_to_push_and_wait(
    tmp_path: Path, monkeypatch
) -> None:
    manager = _build_manager(tmp_path, version="9.9.9")

    observed = {"push": None, "wait": None}

    monkeypatch.setattr(manager, "check_git_status", lambda: True)
    monkeypatch.setattr(manager, "run_tests", lambda: True)
    monkeypatch.setattr(manager, "build_executable", lambda: True)
    monkeypatch.setattr(manager, "test_executable", lambda: True)
    monkeypatch.setattr(
        manager,
        "push_to_github",
        lambda version: observed.__setitem__("push", version) or True,
    )
    monkeypatch.setattr(
        manager,
        "wait_for_github_actions",
        lambda version: observed.__setitem__("wait", version) or True,
    )

    success = manager.release(auto_push=True, message="release note")

    assert success is True
    assert observed == {"push": "9.9.9", "wait": "9.9.9"}


def test_release_skip_flags_and_message_propagate_without_auto_push(
    tmp_path: Path, monkeypatch
) -> None:
    executor = _RecordingLocalExecutor([{"stdout": "master\n"}])
    manager = _build_manager(tmp_path, local_executor=executor)
    vm = _RecordingVM(version="2.0.4")
    manager.vm = vm

    called = {
        "run_tests": False,
        "build_executable": False,
        "test_executable": False,
        "push": False,
        "wait": False,
    }

    monkeypatch.setattr(manager, "check_git_status", lambda: True)
    monkeypatch.setattr(
        manager,
        "run_tests",
        lambda: called.__setitem__("run_tests", True) or True,
    )
    monkeypatch.setattr(
        manager,
        "build_executable",
        lambda: called.__setitem__("build_executable", True) or True,
    )
    monkeypatch.setattr(
        manager,
        "test_executable",
        lambda: called.__setitem__("test_executable", True) or True,
    )
    monkeypatch.setattr(
        manager,
        "push_to_github",
        lambda _version: called.__setitem__("push", True) or True,
    )
    monkeypatch.setattr(
        manager,
        "wait_for_github_actions",
        lambda _version: called.__setitem__("wait", True) or True,
    )

    success = manager.release(
        version_type="minor",
        skip_tests=True,
        skip_build=True,
        auto_push=False,
        message="phase5",
    )

    assert success is True
    assert vm.calls == [("minor", "phase5")]
    assert called == {
        "run_tests": False,
        "build_executable": False,
        "test_executable": False,
        "push": False,
        "wait": False,
    }
    assert executor.calls[0]["cmd"] == ["git", "branch", "--show-current"]


def test_run_tests_uses_injected_python_executable(tmp_path: Path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)

    executor = _RecordingLocalExecutor([{}])
    manager = _build_manager(
        tmp_path,
        local_executor=executor,
        python_executable="/custom/python",
    )

    success = manager.run_tests()

    assert success is True
    assert executor.calls[0]["cmd"] == ["/custom/python", "-m", "pytest", "-v"]


def test_build_executable_uses_injected_python_executable(tmp_path: Path) -> None:
    build_script = tmp_path / "build_exe.py"
    build_script.write_text("print('build')", encoding="utf-8")

    dist_dir = tmp_path / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    exe_file = dist_dir / "integrated_script.exe"
    exe_file.write_bytes(b"exe")

    executor = _RecordingLocalExecutor([{}])
    manager = _build_manager(
        tmp_path,
        local_executor=executor,
        python_executable="/custom/python",
    )

    success = manager.build_executable()

    assert success is True
    assert executor.calls[0]["cmd"] == ["/custom/python", "build_exe.py"]


def test_run_tests_returns_false_when_tests_dir_missing(tmp_path: Path) -> None:
    executor = _RecordingLocalExecutor([])
    manager = _build_manager(tmp_path, local_executor=executor)

    success = manager.run_tests()

    assert success is False
    assert executor.calls == []


def test_run_tests_returns_false_when_pytest_unavailable(tmp_path: Path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)

    executor = _RecordingLocalExecutor([FileNotFoundError("pytest missing")])
    manager = _build_manager(
        tmp_path,
        local_executor=executor,
        python_executable="/custom/python",
    )

    success = manager.run_tests()

    assert success is False
    assert executor.calls[0]["cmd"] == ["/custom/python", "-m", "pytest", "-v"]


def test_test_executable_runs_version_and_help_smoke_checks(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    exe_file = dist_dir / "integrated_script.exe"
    exe_file.write_bytes(b"exe")

    executor = _RecordingLocalExecutor(
        [
            {"returncode": 0, "stdout": "2.0.3\n", "stderr": ""},
            {"returncode": 0, "stdout": "help\n", "stderr": ""},
        ]
    )
    manager = _build_manager(tmp_path, local_executor=executor)

    success = manager.test_executable()

    assert success is True
    assert [call["cmd"] for call in executor.calls] == [
        [str(exe_file), "--version"],
        [str(exe_file), "--help"],
    ]

def test_test_executable_returns_false_when_help_check_fails(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    exe_file = dist_dir / "integrated_script.exe"
    exe_file.write_bytes(b"exe")

    executor = _RecordingLocalExecutor(
        [
            {"returncode": 0, "stdout": "2.0.3\n", "stderr": ""},
            {"returncode": 1, "stdout": "", "stderr": "usage failed"},
        ]
    )
    manager = _build_manager(tmp_path, local_executor=executor)

    success = manager.test_executable()

    assert success is False
    assert [call["cmd"] for call in executor.calls] == [
        [str(exe_file), "--version"],
        [str(exe_file), "--help"],
    ]


def test_test_executable_returns_false_when_version_check_fails(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    exe_file = dist_dir / "integrated_script.exe"
    exe_file.write_bytes(b"exe")

    executor = _RecordingLocalExecutor(
        [
            {"returncode": 1, "stdout": "", "stderr": "version failed"},
        ]
    )
    manager = _build_manager(tmp_path, local_executor=executor)

    success = manager.test_executable()

    assert success is False
    assert [call["cmd"] for call in executor.calls] == [
        [str(exe_file), "--version"],
    ]
