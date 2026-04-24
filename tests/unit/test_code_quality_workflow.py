from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_QUALITY_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "code-quality.yml"


@pytest.mark.unit
def test_code_quality_workflow_keeps_branch_pushes_while_ignoring_release_tags() -> (
    None
):
    content = CODE_QUALITY_WORKFLOW_PATH.read_text(encoding="utf-8")

    required_snippets = [
        "push:",
        "branches:",
        "- '**'",
        "tags-ignore:",
        "- 'v*.*.*'",
        "workflow_dispatch:",
    ]

    for snippet in required_snippets:
        assert snippet in content

    push_index = content.index("push:")
    branches_index = content.index("branches:")
    tags_ignore_index = content.index("tags-ignore:")
    workflow_dispatch_index = content.index("workflow_dispatch:")

    assert push_index < branches_index < tags_ignore_index < workflow_dispatch_index
