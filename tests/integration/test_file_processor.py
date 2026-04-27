from contextlib import contextmanager
from pathlib import Path

from integrated_script.processors.file_processor import FileProcessor


def test_copy_files_recursive_preserves_structure(tmp_path: Path) -> None:
    source = tmp_path / "source"
    nested = source / "nested"
    nested.mkdir(parents=True)
    (nested / "a.txt").write_text("alpha", encoding="utf-8")

    target = tmp_path / "target"
    processor = FileProcessor()

    result = processor.copy_files(
        str(source),
        str(target),
        recursive=True,
        preserve_structure=True,
    )

    assert result["success"] is True
    assert result["statistics"]["copied_count"] == 1
    assert (target / "nested" / "a.txt").exists()


def test_move_images_by_count_moves_requested_number(tmp_path: Path) -> None:
    source = tmp_path / "images"
    source.mkdir(parents=True)
    for name in ["001.jpg", "002.jpg", "003.jpg"]:
        (source / name).write_text("img", encoding="utf-8")

    target = tmp_path / "moved"
    processor = FileProcessor()

    result = processor.move_images_by_count(str(source), str(target), count=2)

    assert result["success"] is True
    assert result["statistics"]["moved_count"] == 2
    assert len(list(target.glob("*.jpg"))) == 2


def test_move_images_by_count_reports_progress_by_requested_move_count(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "images"
    source.mkdir(parents=True)
    for name in ["001.jpg", "002.jpg", "003.jpg"]:
        (source / name).write_text("img", encoding="utf-8")

    target = tmp_path / "moved"
    processor = FileProcessor()
    captured: dict[str, object] = {}
    updates: list[int] = []

    @contextmanager
    def _fake_progress_context(total, description, show_progress=True, **_kwargs):
        captured["total"] = total
        captured["description"] = description
        captured["show_progress"] = show_progress

        class _Manager:
            def update_progress(self, n=1, description=None):
                del description
                updates.append(n)

        yield _Manager()

    monkeypatch.setattr(
        "integrated_script.processors.file.core.progress_context",
        _fake_progress_context,
    )

    result = processor.move_images_by_count(str(source), str(target), count=2)

    assert result["success"] is True
    assert captured == {
        "total": 2,
        "description": "按数量移动图片",
        "show_progress": True,
    }
    assert updates == [1, 1]


def test_move_images_by_count_translates_9999_to_all_candidates_for_progress(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "images"
    source.mkdir(parents=True)
    for name in ["001.jpg", "002.jpg", "003.jpg"]:
        (source / name).write_text("img", encoding="utf-8")

    target = tmp_path / "moved"
    processor = FileProcessor()
    captured_totals: list[int] = []
    updates: list[int] = []

    @contextmanager
    def _fake_progress_context(total, description, show_progress=True, **_kwargs):
        del description, show_progress
        captured_totals.append(total)

        class _Manager:
            def update_progress(self, n=1, description=None):
                del description
                updates.append(n)

        yield _Manager()

    monkeypatch.setattr(
        "integrated_script.processors.file.core.progress_context",
        _fake_progress_context,
    )

    result = processor.move_images_by_count(str(source), str(target), count=9999)

    assert result["success"] is True
    assert result["statistics"]["moved_count"] == 3
    assert captured_totals == [3]
    assert updates == [1, 1, 1]


def test_move_images_by_count_does_not_advance_progress_on_failed_move(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "images"
    source.mkdir(parents=True)
    for name in ["001.jpg", "002.jpg", "003.jpg"]:
        (source / name).write_text("img", encoding="utf-8")

    target = tmp_path / "moved"
    processor = FileProcessor()
    updates: list[int] = []

    @contextmanager
    def _fake_progress_context(total, description, show_progress=True, **_kwargs):
        del total, description, show_progress

        class _Manager:
            def update_progress(self, n=1, description=None):
                del description
                updates.append(n)

        yield _Manager()

    def _fake_move_file_safe(source_path, target_path):
        if Path(source_path).name == "001.jpg":
            raise OSError("inject-move-failure")
        Path(target_path).write_text(Path(source_path).read_text(encoding="utf-8"))
        Path(source_path).unlink()

    monkeypatch.setattr(
        "integrated_script.processors.file.core.progress_context",
        _fake_progress_context,
    )
    monkeypatch.setattr(
        "integrated_script.processors.file.core.move_file_safe",
        _fake_move_file_safe,
    )

    result = processor.move_images_by_count(str(source), str(target), count=2)

    assert result["success"] is True
    assert result["statistics"]["moved_count"] == 2
    assert result["statistics"]["failed_count"] == 1
    assert len(result["failed_files"]) == 1
    assert updates == [1, 1]


def test_move_images_by_count_respects_ui_show_progress_false(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "images"
    source.mkdir(parents=True)
    for name in ["001.jpg", "002.jpg"]:
        (source / name).write_text("img", encoding="utf-8")

    target = tmp_path / "moved"
    processor = FileProcessor()
    captured_show_progress: list[bool] = []

    @contextmanager
    def _fake_progress_context(total, description, show_progress=True, **_kwargs):
        del total, description
        captured_show_progress.append(show_progress)

        class _Manager:
            def update_progress(self, n=1, description=None):
                del n, description
                return None

        yield _Manager()

    monkeypatch.setattr(
        "integrated_script.processors.file.core.progress_context",
        _fake_progress_context,
    )
    monkeypatch.setattr(processor, "get_config", lambda key, default=None: False)

    result = processor.move_images_by_count(str(source), str(target), count=1)

    assert result["success"] is True
    assert captured_show_progress == [False]


def test_rename_files_with_temp_preview_only_does_not_change_files(
    tmp_path: Path,
) -> None:
    target = tmp_path / "rename"
    target.mkdir(parents=True)
    (target / "a.txt").write_text("a", encoding="utf-8")
    (target / "b.txt").write_text("b", encoding="utf-8")

    processor = FileProcessor()
    result = processor.rename_files_with_temp(
        str(target),
        "sample_{index}{ext}",
        file_patterns=["*.txt"],
        preview_only=True,
    )

    assert result["success"] is True
    assert result["statistics"]["total_files"] == 2
    assert (target / "a.txt").exists()
    assert (target / "b.txt").exists()


def test_rename_images_labels_sync_keeps_stems_consistent_on_name_conflict(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)

    (images / "car.jpg").write_text("img", encoding="utf-8")
    (labels / "car.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    # 预置同名标签冲突，验证同步重命名仍保持 image/label 同 stem
    (labels / "frame_001.txt").write_text("0 0.1 0.1 0.1 0.1\n", encoding="utf-8")

    processor = FileProcessor()
    result = processor.rename_images_labels_sync(
        str(images),
        str(labels),
        prefix="frame",
        digits=3,
        shuffle_order=False,
    )

    assert result["success"] is True
    assert result["statistics"]["renamed_count"] == 1

    renamed = result["renamed_pairs"][0]
    new_img = renamed["new_img"]
    new_label = renamed["new_label"]

    assert Path(new_img).stem == Path(new_label).stem
    assert (images / new_img).exists()
    assert (labels / new_label).exists()


def test_rename_images_labels_sync_rolls_back_when_temp_label_rename_fails(
    tmp_path: Path, monkeypatch
) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)

    original_img = images / "bike.jpg"
    original_label = labels / "bike.txt"
    original_img.write_text("img", encoding="utf-8")
    original_label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    processor = FileProcessor()

    real_rename = Path.rename

    def _patched_rename(self: Path, target):
        target_path = Path(target)
        # 第一阶段：只在标签改为 temp_label_* 时注入失败
        if self.name == "bike.txt" and target_path.name.startswith("temp_label_"):
            raise OSError("inject-temp-label-rename-failure")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", _patched_rename)

    result = processor.rename_images_labels_sync(
        str(images),
        str(labels),
        prefix="frame",
        digits=3,
        shuffle_order=False,
    )

    assert result["success"] is False
    assert result["statistics"]["failed_count"] == 1
    assert result["statistics"]["renamed_count"] == 0

    # 第一阶段失败后，两侧都应回滚为原始文件名
    assert (images / "bike.jpg").exists()
    assert (labels / "bike.txt").exists()


def test_rename_files_with_temp_rolls_back_all_when_first_stage_fails_mid_batch(
    tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "rename"
    target.mkdir(parents=True)
    (target / "a.txt").write_text("a", encoding="utf-8")
    (target / "b.txt").write_text("b", encoding="utf-8")

    processor = FileProcessor()
    real_rename = Path.rename

    def _patched_rename(self: Path, target_path):
        target_path = Path(target_path)
        if self.name == "b.txt" and target_path.name.startswith("temp_"):
            raise OSError("inject-first-stage-batch-failure")
        return real_rename(self, target_path)

    monkeypatch.setattr(Path, "rename", _patched_rename)

    result = processor.rename_files_with_temp(
        str(target),
        "sample_{index}{ext}",
        file_patterns=["*.txt"],
    )

    assert result["success"] is False
    assert result["statistics"]["renamed_count"] == 0
    assert result["statistics"]["failed_count"] >= 1
    assert (target / "a.txt").exists()
    assert (target / "b.txt").exists()
    assert not any(target.glob("temp_*"))


def test_rename_files_with_temp_rolls_back_all_when_second_stage_fails_mid_batch(
    tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "rename"
    target.mkdir(parents=True)
    (target / "a.txt").write_text("a", encoding="utf-8")
    (target / "b.txt").write_text("b", encoding="utf-8")

    processor = FileProcessor()
    real_rename = Path.rename

    def _patched_rename(self: Path, target_path):
        target_path = Path(target_path)
        if self.name.startswith("temp_") and target_path.name == "sample_2.txt":
            raise OSError("inject-second-stage-batch-failure")
        return real_rename(self, target_path)

    monkeypatch.setattr(Path, "rename", _patched_rename)

    result = processor.rename_files_with_temp(
        str(target),
        "sample_{index}{ext}",
        file_patterns=["*.txt"],
    )

    assert result["success"] is False
    assert result["statistics"]["renamed_count"] == 0
    assert result["statistics"]["failed_count"] >= 1
    assert (target / "a.txt").exists()
    assert (target / "b.txt").exists()
    assert not any(target.glob("temp_*"))
    assert not (target / "sample_1.txt").exists()
    assert not (target / "sample_2.txt").exists()


def test_rename_images_labels_sync_rolls_back_all_when_second_stage_label_rename_fails_mid_batch(
    tmp_path: Path, monkeypatch
) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)

    (images / "a.jpg").write_text("img", encoding="utf-8")
    (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (images / "b.jpg").write_text("img", encoding="utf-8")
    (labels / "b.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    processor = FileProcessor()
    real_rename = Path.rename

    def _patched_rename(self: Path, target_path):
        target_path = Path(target_path)
        if self.name.startswith("temp_label_") and target_path.name == "frame_002.txt":
            raise OSError("inject-sync-final-label-failure")
        return real_rename(self, target_path)

    monkeypatch.setattr(Path, "rename", _patched_rename)

    result = processor.rename_images_labels_sync(
        str(images),
        str(labels),
        prefix="frame",
        digits=3,
        shuffle_order=False,
    )

    assert result["success"] is False
    assert result["statistics"]["renamed_count"] == 0
    assert result["statistics"]["failed_count"] >= 1

    # 第二阶段中途失败后，全部应回滚到原始名称
    assert (images / "a.jpg").exists()
    assert (labels / "a.txt").exists()
    assert (images / "b.jpg").exists()
    assert (labels / "b.txt").exists()


def test_rename_files_with_temp_rejects_escape_pattern(tmp_path: Path) -> None:
    target = tmp_path / "rename"
    target.mkdir(parents=True)
    (target / "a.txt").write_text("a", encoding="utf-8")

    processor = FileProcessor()
    result = processor.rename_files_with_temp(
        str(target),
        "../escape_{index}{ext}",
        file_patterns=["*.txt"],
    )

    assert result["success"] is False
    assert result["statistics"]["renamed_count"] == 0
    assert (target / "a.txt").exists()
    assert not (tmp_path / "escape_1.txt").exists()


def test_rename_images_labels_sync_rejects_escape_prefix(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)

    (images / "a.jpg").write_text("img", encoding="utf-8")
    (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    processor = FileProcessor()
    result = processor.rename_images_labels_sync(
        str(images),
        str(labels),
        prefix="../frame",
        digits=3,
        shuffle_order=False,
    )

    assert result["success"] is False
    assert result["statistics"]["renamed_count"] == 0
    assert (images / "a.jpg").exists()
    assert (labels / "a.txt").exists()
    assert not (tmp_path / "frame_001.jpg").exists()
    assert not (tmp_path / "frame_001.txt").exists()
