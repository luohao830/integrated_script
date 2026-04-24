from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_COMMAND_PATH = REPO_ROOT / ".claude" / "commands" / "release.md"
RELEASE_SKILL_PATH = REPO_ROOT / ".claude" / "skills" / "release" / "SKILL.md"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("doc_path", "required_snippets"),
    [
        (
            RELEASE_COMMAND_PATH,
            [
                "版本基线使用当前分支可达的上一个语义化 tag",
                "版本递增的唯一基线是当前分支上一个语义化 tag；`pyproject.toml` 只是在确认发布后被更新为目标版本，不参与版本计算、基线推导或首发版本判定。",
                "若没有历史 tag，则按初始发布处理，并要求用户确认首个版本号",
            ],
        ),
        (
            RELEASE_SKILL_PATH,
            [
                "版本基线应直接来自 git tag，而不是 `pyproject.toml` 或运行时包元数据；这样才能与实际发布链路保持一致。若没有历史 tag，则按初始发布处理，并要求用户确认首个版本号。",
                "当前版本基线（从上一个 tag 读取）",
            ],
        ),
    ],
)
def test_release_docs_use_git_tag_as_only_version_baseline(
    doc_path: Path,
    required_snippets: list[str],
) -> None:
    content = doc_path.read_text(encoding="utf-8")

    for snippet in required_snippets:
        assert snippet in content


@pytest.mark.unit
@pytest.mark.parametrize("doc_path", [RELEASE_COMMAND_PATH, RELEASE_SKILL_PATH])
def test_release_docs_do_not_use_pyproject_or_runtime_metadata_as_version_baseline(
    doc_path: Path,
) -> None:
    content = doc_path.read_text(encoding="utf-8")

    assert "版本从当前版本递增" not in content
    assert "运行时包元数据" not in content or "而不是 `pyproject.toml` 或运行时包元数据" in content


def test_release_command_doc_requires_manual_confirmation_for_first_version_without_tags() -> None:
    content = RELEASE_COMMAND_PATH.read_text(encoding="utf-8")

    assert "若没有历史 tag，则按初始发布处理，并要求用户确认首个版本号" in content
    assert "不得回退到 `pyproject.toml` 当前版本作为计算基线" in content
    assert "不得回退到 `pyproject.toml` 当前版本作为发布基线" in content


@pytest.mark.unit
@pytest.mark.parametrize("doc_path", [RELEASE_COMMAND_PATH, RELEASE_SKILL_PATH])
def test_release_docs_require_commit_before_tag_and_push(doc_path: Path) -> None:
    content = doc_path.read_text(encoding="utf-8")

    required_snippets = [
        "执行顺序必须固定为",
        "1. 更新 `pyproject.toml` 的 `[project].version`",
        "2. 提交版本变更",
        "3. 创建 annotated tag",
        "4. `git push origin <current-branch>`",
        "5. `git push origin v<next_version>`",
        "修改 pyproject.toml 后，必须先提交版本变更，再创建 tag 和执行 push。",
        "版本更新提交成功之前，不得进入后续发布动作。",
        "不得在未提交 pyproject.toml 变更前创建 annotated tag。",
        "不得在未提交版本变更前执行任何 push。",
    ]

    for snippet in required_snippets:
        assert snippet in content

    assert content.index("1. 更新 `pyproject.toml` 的 `[project].version`") < content.index("2. 提交版本变更")
    assert content.index("2. 提交版本变更") < content.index("3. 创建 annotated tag")
    assert content.index("3. 创建 annotated tag") < content.index("4. `git push origin <current-branch>`")
    assert content.index("4. `git push origin <current-branch>`") < content.index("5. `git push origin v<next_version>`")
