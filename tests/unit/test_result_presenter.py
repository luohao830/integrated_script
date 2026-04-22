from typing import List

from integrated_script.ui.presenters.result_presenter import render_result


def test_render_result_prefers_error_field_on_failure(monkeypatch) -> None:
    printed: List[str] = []
    monkeypatch.setattr(
        "builtins.print",
        lambda *args, **_kwargs: printed.append(" ".join(str(arg) for arg in args)),
    )

    render_result(
        {"success": False, "error": "来自 error 字段", "message": "来自 message 字段"}
    )

    assert any("错误信息: 来自 error 字段" in line for line in printed)


def test_render_result_prints_statistics_adjusted_path_hint(monkeypatch) -> None:
    printed: List[str] = []
    monkeypatch.setattr(
        "builtins.print",
        lambda *args, **_kwargs: printed.append(" ".join(str(arg) for arg in args)),
    )

    render_result(
        {
            "success": True,
            "statistics": {
                "dataset_path": "/tmp/dataset",
                "original_path": "/tmp/dataset/images",
                "is_valid": True,
                "orphaned_images": 0,
            },
        }
    )

    assert any("统计信息:" in line for line in printed)
    assert any("数据集路径: /tmp/dataset" in line for line in printed)
    assert any("已自动调整为数据集根目录" in line for line in printed)


def test_render_result_prints_failed_pairs_details(monkeypatch) -> None:
    printed: List[str] = []
    monkeypatch.setattr(
        "builtins.print",
        lambda *args, **_kwargs: printed.append(" ".join(str(arg) for arg in args)),
    )

    render_result(
        {
            "success": False,
            "error": "批处理失败",
            "failed_pairs": [
                {
                    "img_file": "a.jpg",
                    "label_file": "a.txt",
                    "error": "copy failed",
                    "action": "copy",
                }
            ],
        }
    )

    assert any("失败文件详情:" in line for line in printed)
    assert any("图像文件: a.jpg" in line for line in printed)
    assert any("标签文件: a.txt" in line for line in printed)
    assert any("失败原因: copy failed" in line for line in printed)
