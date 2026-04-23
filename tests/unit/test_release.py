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

    def _mark_run_tests() -> bool:
        called["run_tests"] = True
        return True

    def _mark_build_executable() -> bool:
        called["build_executable"] = True
        return True

    def _mark_test_executable() -> bool:
        called["test_executable"] = True
        return True

    def _mark_vm_release(*_args) -> str:
        called["vm_release"] = True
        return "2.0.4"

    monkeypatch.setattr(manager, "check_git_status", lambda: False)
    monkeypatch.setattr(manager, "run_tests", _mark_run_tests)
    monkeypatch.setattr(manager, "build_executable", _mark_build_executable)
    monkeypatch.setattr(manager, "test_executable", _mark_test_executable)
    monkeypatch.setattr(manager.vm, "release", _mark_vm_release)

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

    def _record_push(version: str) -> bool:
        observed["push"] = version
        return True

    def _record_wait(version: str) -> bool:
        observed["wait"] = version
        return True

    monkeypatch.setattr(manager, "check_git_status", lambda: True)
    monkeypatch.setattr(manager, "run_tests", lambda: True)
    monkeypatch.setattr(manager, "build_executable", lambda: True)
    monkeypatch.setattr(manager, "test_executable", lambda: True)
    monkeypatch.setattr(manager, "push_to_github", _record_push)
    monkeypatch.setattr(manager, "wait_for_github_actions", _record_wait)

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

    def _mark_run_tests() -> bool:
        called["run_tests"] = True
        return True

    def _mark_build_executable() -> bool:
        called["build_executable"] = True
        return True

    def _mark_test_executable() -> bool:
        called["test_executable"] = True
        return True

    def _mark_push(_version: str) -> bool:
        called["push"] = True
        return True

    def _mark_wait(_version: str) -> bool:
        called["wait"] = True
        return True

    monkeypatch.setattr(manager, "check_git_status", lambda: True)
    monkeypatch.setattr(manager, "run_tests", _mark_run_tests)
    monkeypatch.setattr(manager, "build_executable", _mark_build_executable)
    monkeypatch.setattr(manager, "test_executable", _mark_test_executable)
    monkeypatch.setattr(manager, "push_to_github", _mark_push)
    monkeypatch.setattr(manager, "wait_for_github_actions", _mark_wait)

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


def test_github_actions_client_uses_default_api_url_when_no_override(
    monkeypatch,
) -> None:
    monkeypatch.delenv("INTEGRATED_SCRIPT_GITHUB_ACTIONS_API_URL", raising=False)

    client = release_module.GitHubActionsClient()

    assert (
        client.api_url
        == "https://api.github.com/repos/luohao091/integrated_script/actions/runs"
    )


def test_github_actions_client_uses_env_api_url_when_set(monkeypatch) -> None:
    monkeypatch.setenv(
        "INTEGRATED_SCRIPT_GITHUB_ACTIONS_API_URL",
        "https://api.github.com/repos/acme/demo/actions/runs",
    )

    client = release_module.GitHubActionsClient()

    assert client.api_url == "https://api.github.com/repos/acme/demo/actions/runs"


def test_github_actions_client_prefers_explicit_api_url_over_env(monkeypatch) -> None:
    monkeypatch.setenv(
        "INTEGRATED_SCRIPT_GITHUB_ACTIONS_API_URL",
        "https://api.github.com/repos/acme/demo/actions/runs",
    )

    client = release_module.GitHubActionsClient(
        api_url="https://api.github.com/repos/owner/explicit/actions/runs"
    )

    assert client.api_url == "https://api.github.com/repos/owner/explicit/actions/runs"


def test_main_passes_cli_github_api_url_to_release_manager(monkeypatch) -> None:
    observed: dict = {}

    class _FakeReleaseManager:
        def __init__(self, **kwargs):
            observed["init_kwargs"] = kwargs

        def release(
            self,
            version_type: str,
            skip_tests: bool,
            skip_build: bool,
            auto_push: bool,
            message,
            release_overview,
        ) -> bool:
            observed["release_kwargs"] = {
                "version_type": version_type,
                "skip_tests": skip_tests,
                "skip_build": skip_build,
                "auto_push": auto_push,
                "message": message,
                "release_overview": release_overview,
            }
            return True

    monkeypatch.setattr(release_module, "ReleaseManager", _FakeReleaseManager)
    monkeypatch.setattr(
        release_module.sys,
        "argv",
        [
            "release.py",
            "minor",
            "--skip-tests",
            "--skip-build",
            "--auto-push",
            "--github-actions-api-url",
            "https://api.github.com/repos/acme/demo/actions/runs",
            "-m",
            "p2",
            "--release-overview",
            "## 发布概况\n- 变更A",
        ],
    )

    release_module.main()

    assert observed["init_kwargs"]["github_actions_api_url"] == (
        "https://api.github.com/repos/acme/demo/actions/runs"
    )
    assert observed["release_kwargs"] == {
        "version_type": "minor",
        "skip_tests": True,
        "skip_build": True,
        "auto_push": True,
        "message": "p2",
        "release_overview": "## 发布概况\n- 变更A",
    }


def test_main_reads_release_overview_from_file(tmp_path: Path, monkeypatch) -> None:
    observed: dict = {}
    overview_file = tmp_path / "overview.md"
    overview_file.write_text("## 发布概况\n- 文件输入", encoding="utf-8")

    class _FakeReleaseManager:
        def __init__(self, **kwargs):
            observed["init_kwargs"] = kwargs

        def release(
            self,
            version_type: str,
            skip_tests: bool,
            skip_build: bool,
            auto_push: bool,
            message,
            release_overview,
        ) -> bool:
            observed["release_kwargs"] = {
                "version_type": version_type,
                "skip_tests": skip_tests,
                "skip_build": skip_build,
                "auto_push": auto_push,
                "message": message,
                "release_overview": release_overview,
            }
            return True

    monkeypatch.setattr(release_module, "ReleaseManager", _FakeReleaseManager)
    monkeypatch.setattr(
        release_module.sys,
        "argv",
        [
            "release.py",
            "patch",
            "--release-overview-file",
            str(overview_file),
        ],
    )

    release_module.main()

    assert observed["release_kwargs"] == {
        "version_type": "patch",
        "skip_tests": False,
        "skip_build": False,
        "auto_push": False,
        "message": None,
        "release_overview": "## 发布概况\n- 文件输入",
    }


def test_release_appends_overview_markers_to_tag_message(tmp_path: Path, monkeypatch) -> None:
    executor = _RecordingLocalExecutor([{"stdout": "master\n"}])
    manager = _build_manager(tmp_path, local_executor=executor)
    vm = _RecordingVM(version="2.0.4")
    manager.vm = vm

    monkeypatch.setattr(manager, "check_git_status", lambda: True)

    success = manager.release(
        version_type="patch",
        skip_tests=True,
        skip_build=True,
        auto_push=False,
        message="manual",
        release_overview="## 发布概况\n- 新增A",
    )

    assert success is True
    assert vm.calls == [
        (
            "patch",
            "manual\n\n"
            "<!-- release-overview:start -->\n"
            "## 发布概况\n- 新增A\n"
            "<!-- release-overview:end -->",
        )
    ]


def test_append_release_overview_handles_empty_inputs() -> None:
    assert release_module.append_release_overview(None, None) is None
    assert release_module.append_release_overview("base", None) == "base"
    assert (
        release_module.append_release_overview(None, "概况")
        == "<!-- release-overview:start -->\n概况\n<!-- release-overview:end -->"
    )
