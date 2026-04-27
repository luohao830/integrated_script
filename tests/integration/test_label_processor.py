from pathlib import Path

from integrated_script.processors.label_processor import LabelProcessor


def test_create_empty_labels_creates_for_all_images(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)

    (images / "a.jpg").write_text("img", encoding="utf-8")
    (images / "b.jpg").write_text("img", encoding="utf-8")

    processor = LabelProcessor()
    result = processor.create_empty_labels(str(images), str(labels), overwrite=False)

    assert result["success"] is True
    assert result["statistics"]["created_count"] == 2
    assert (labels / "a.txt").exists()
    assert (labels / "b.txt").exists()


def test_flip_labels_horizontal_updates_x_center(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    labels.mkdir(parents=True)
    label_file = labels / "a.txt"
    label_file.write_text("0 0.200000 0.300000 0.100000 0.100000\n", encoding="utf-8")

    processor = LabelProcessor()
    result = processor.flip_labels(str(labels), flip_type="horizontal", backup=False)

    assert result["success"] is True
    content = label_file.read_text(encoding="utf-8").strip()
    assert content.startswith("0 0.800000 0.300000")


def test_filter_labels_by_class_keep_removes_other_classes(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    labels.mkdir(parents=True)
    label_file = labels / "a.txt"
    label_file.write_text(
        "0 0.5 0.5 0.1 0.1\n1 0.4 0.4 0.2 0.2\n",
        encoding="utf-8",
    )

    processor = LabelProcessor()
    result = processor.filter_labels_by_class(
        str(labels),
        target_classes=[0],
        action="keep",
        backup=False,
    )

    assert result["success"] is True
    content = label_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 1
    assert content[0].startswith("0 ")


def test_remove_empty_labels_and_images_removes_pair(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    labels = dataset / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)

    (images / "x.jpg").write_text("img", encoding="utf-8")
    (labels / "x.txt").write_text("", encoding="utf-8")

    processor = LabelProcessor()
    result = processor.remove_empty_labels_and_images(str(dataset))

    assert result["success"] is True
    assert result["statistics"]["removed_labels"] == 1
    assert not (labels / "x.txt").exists()
    assert not (images / "x.jpg").exists()


def test_filter_labels_by_class_keeps_invalid_class_line_as_is(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    labels.mkdir(parents=True)
    label_file = labels / "mixed.txt"
    label_file.write_text(
        "x 0.5 0.5 0.1 0.1\n1 0.4 0.4 0.2 0.2\n",
        encoding="utf-8",
    )

    processor = LabelProcessor()
    result = processor.filter_labels_by_class(
        str(labels),
        target_classes=[1],
        action="keep",
        backup=False,
    )

    assert result["success"] is True
    lines = label_file.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "x 0.5 0.5 0.1 0.1"
    assert lines[1].startswith("1 ")


def test_remove_labels_with_only_class_excludes_classes_txt_from_statistics(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    labels = dataset / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)

    (images / "a.jpg").write_text("img", encoding="utf-8")
    (labels / "a.txt").write_text("1 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (labels / "classes.txt").write_text("class-a\n", encoding="utf-8")

    processor = LabelProcessor()
    result = processor.remove_labels_with_only_class(str(dataset), target_class=1)

    assert result["success"] is True
    assert result["statistics"]["total_labels"] == 1
    assert result["statistics"]["target_class_only_labels"] == 1
    assert result["statistics"]["removed_labels"] == 1
    assert result["statistics"]["removed_images"] == 1
    assert result["statistics"]["failed_count"] == 0
    assert (labels / "classes.txt").exists()
    assert not (labels / "a.txt").exists()
    assert not (images / "a.jpg").exists()
    assert all(
        Path(pair["label_file"]).name != "classes.txt"
        for pair in result["removed_pairs"]
    )
