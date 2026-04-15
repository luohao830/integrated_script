from pathlib import Path
from types import ModuleType
import sys

import pytest

from integrated_script.config.exceptions import FileProcessingError, PathError
from integrated_script.core.utils import (
    copy_file_safe,
    create_directory,
    cv2_imread_unicode,
    cv2_imwrite_unicode,
    delete_file_safe,
    format_file_size,
    get_file_list,
    get_unique_filename,
    move_file_safe,
    safe_file_operation,
    validate_path,
)


class _EncodedBytes:
    def __init__(self, payload: bytes):
        self._payload = payload

    def tobytes(self) -> bytes:
        return self._payload


class _FakeCv2:
    IMREAD_COLOR = 1

    def __init__(self, should_encode: bool = True):
        self.decoded_input = None
        self.decoded_flags = None
        self.should_encode = should_encode

    def imdecode(self, array, flags):
        self.decoded_input = array
        self.decoded_flags = flags
        return {"image": True, "flags": flags}

    def imencode(self, suffix, _image, _params):
        if not self.should_encode:
            return False, None
        return True, _EncodedBytes(f"{suffix}".encode("utf-8"))


def _install_fake_cv_modules(monkeypatch, fake_cv2: _FakeCv2) -> None:
    numpy_module = ModuleType("numpy")
    numpy_module.uint8 = object()
    numpy_module.frombuffer = lambda raw, dtype=None: {"raw": raw, "dtype": dtype}

    cv2_module = ModuleType("cv2")
    cv2_module.IMREAD_COLOR = fake_cv2.IMREAD_COLOR
    cv2_module.imdecode = fake_cv2.imdecode
    cv2_module.imencode = fake_cv2.imencode

    monkeypatch.setitem(sys.modules, "numpy", numpy_module)
    monkeypatch.setitem(sys.modules, "cv2", cv2_module)


def test_validate_path_raises_on_empty_path() -> None:
    with pytest.raises(PathError):
        validate_path("")


def test_validate_path_creates_directory_when_requested(tmp_path: Path) -> None:
    target = tmp_path / "new-dir"

    created = validate_path(
        target,
        must_exist=True,
        must_be_dir=True,
        create_if_missing=True,
    )

    assert created.exists()
    assert created.is_dir()


def test_validate_path_returns_path_when_not_required_to_exist(
    tmp_path: Path,
) -> None:
    target = tmp_path / "not-exist"

    result = validate_path(target, must_exist=False)

    assert result == target.resolve()


def test_validate_path_raises_when_directory_creation_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = tmp_path / "will-fail"

    def _raise_mkdir(*_args, **_kwargs):
        raise OSError("mkdir failed")

    monkeypatch.setattr(Path, "mkdir", _raise_mkdir)

    with pytest.raises(PathError):
        validate_path(
            target,
            must_exist=True,
            must_be_dir=True,
            create_if_missing=True,
        )


def test_get_file_list_filters_extensions_and_hidden_files(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.jpg").write_text("b", encoding="utf-8")
    (tmp_path / ".hidden.txt").write_text("h", encoding="utf-8")

    files = get_file_list(
        tmp_path,
        extensions=[".txt"],
        recursive=False,
        include_hidden=False,
    )

    assert [p.name for p in files] == ["a.txt"]


def test_get_file_list_includes_hidden_when_requested(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / ".hidden.txt").write_text("h", encoding="utf-8")

    files = get_file_list(tmp_path, extensions=[".txt"], include_hidden=True)

    assert sorted(p.name for p in files) == [".hidden.txt", "a.txt"]


def test_get_file_list_wraps_iteration_errors(tmp_path: Path, monkeypatch) -> None:
    def _raise_glob(*_args, **_kwargs):
        raise OSError("glob failed")

    monkeypatch.setattr(Path, "glob", _raise_glob)

    with pytest.raises(FileProcessingError):
        get_file_list(tmp_path)


def test_get_unique_filename_returns_original_when_unused(tmp_path: Path) -> None:
    unique_path = get_unique_filename(tmp_path, "demo.txt")

    assert unique_path.name == "demo.txt"


def test_get_unique_filename_returns_incremented_name(tmp_path: Path) -> None:
    (tmp_path / "demo.txt").write_text("1", encoding="utf-8")

    unique_path = get_unique_filename(tmp_path, "demo.txt")

    assert unique_path.name == "demo_1.txt"


def test_get_unique_filename_skips_existing_suffixes(tmp_path: Path) -> None:
    (tmp_path / "demo.txt").write_text("1", encoding="utf-8")
    (tmp_path / "demo_1.txt").write_text("1", encoding="utf-8")

    unique_path = get_unique_filename(tmp_path, "demo.txt")

    assert unique_path.name == "demo_2.txt"


def test_create_copy_move_and_delete_file_operations(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("data", encoding="utf-8")

    output_dir = create_directory(tmp_path / "out")
    copied = copy_file_safe(source, output_dir / "copied.txt")
    moved = move_file_safe(copied, output_dir / "nested" / "moved.txt")

    assert output_dir.is_dir()
    assert copied.exists() is False
    assert moved.exists()
    assert moved.read_text(encoding="utf-8") == "data"


def test_delete_file_safe_handles_missing_file(tmp_path: Path) -> None:
    deleted = delete_file_safe(tmp_path / "missing.txt", missing_ok=True)

    assert deleted is False


def test_delete_file_safe_raises_when_missing_not_allowed(tmp_path: Path) -> None:
    with pytest.raises(FileProcessingError):
        delete_file_safe(tmp_path / "missing.txt", missing_ok=False)


def test_delete_file_safe_deletes_directory(tmp_path: Path) -> None:
    target_dir = tmp_path / "to-delete"
    target_dir.mkdir()
    (target_dir / "a.txt").write_text("x", encoding="utf-8")

    deleted = delete_file_safe(target_dir)

    assert deleted is True
    assert target_dir.exists() is False


def test_safe_file_operation_wraps_permission_error() -> None:
    @safe_file_operation("dummy")
    def _op(*_args, **_kwargs):
        raise PermissionError("denied")

    with pytest.raises(FileProcessingError) as exc_info:
        _op("/tmp/a")

    assert exc_info.value.operation == "dummy"
    assert exc_info.value.file_path == "/tmp/a"


def test_safe_file_operation_wraps_os_error() -> None:
    @safe_file_operation("dummy")
    def _op(*_args, **_kwargs):
        raise OSError("os-failed")

    with pytest.raises(FileProcessingError) as exc_info:
        _op("/tmp/b")

    assert exc_info.value.operation == "dummy"
    assert exc_info.value.file_path == "/tmp/b"


def test_safe_file_operation_wraps_unknown_error() -> None:
    @safe_file_operation("dummy")
    def _op(*_args, **_kwargs):
        raise RuntimeError("unknown")

    with pytest.raises(FileProcessingError) as exc_info:
        _op("/tmp/c")

    assert exc_info.value.operation == "dummy"
    assert exc_info.value.file_path == "/tmp/c"


def test_cv2_imread_unicode_returns_none_on_read_error(monkeypatch) -> None:
    fake_cv2 = _FakeCv2()
    _install_fake_cv_modules(monkeypatch, fake_cv2)

    result = cv2_imread_unicode("/not/exist.jpg")

    assert result is None


def test_cv2_imread_unicode_uses_default_flag(tmp_path: Path, monkeypatch) -> None:
    fake_cv2 = _FakeCv2()
    _install_fake_cv_modules(monkeypatch, fake_cv2)

    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"raw-image")

    result = cv2_imread_unicode(image_path)

    assert result["image"] is True
    assert fake_cv2.decoded_flags == fake_cv2.IMREAD_COLOR


def test_cv2_imwrite_unicode_returns_false_when_encode_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_cv2 = _FakeCv2(should_encode=False)
    _install_fake_cv_modules(monkeypatch, fake_cv2)

    image_path = tmp_path / "output.jpg"

    result = cv2_imwrite_unicode(image_path, image={"x": 1})

    assert result is False
    assert image_path.exists() is False


def test_cv2_imwrite_unicode_writes_encoded_bytes(tmp_path: Path, monkeypatch) -> None:
    fake_cv2 = _FakeCv2(should_encode=True)
    _install_fake_cv_modules(monkeypatch, fake_cv2)

    image_path = tmp_path / "output"

    result = cv2_imwrite_unicode(image_path, image={"x": 1})

    assert result is True
    assert image_path.read_bytes() == b".png"


def test_format_file_size_formats_boundaries() -> None:
    assert format_file_size(0) == "0 B"
    assert format_file_size(1024) == "1.0 KB"
