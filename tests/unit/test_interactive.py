import pytest

from integrated_script.ui.interactive import InteractiveInterface


@pytest.fixture
def interface_with_non_exe(monkeypatch):
    monkeypatch.setattr(InteractiveInterface, "_is_running_as_exe", lambda self: False)
    return InteractiveInterface()


@pytest.fixture
def interface_with_exe(monkeypatch):
    monkeypatch.setattr(InteractiveInterface, "_is_running_as_exe", lambda self: True)
    return InteractiveInterface()


def test_main_menu_includes_environment_entry_when_not_exe(interface_with_non_exe) -> None:
    options = [name for name, _ in interface_with_non_exe.menu_system.main_menu["options"]]

    assert "环境检查与配置" in options


def test_main_menu_hides_environment_entry_when_exe(interface_with_exe) -> None:
    options = [name for name, _ in interface_with_exe.menu_system.main_menu["options"]]

    assert "环境检查与配置" not in options


def test_get_processor_reuses_cached_instance(interface_with_non_exe) -> None:
    processor1 = interface_with_non_exe._get_processor("file")
    processor2 = interface_with_non_exe._get_processor("file")

    assert processor1 is processor2


def test_get_processor_unknown_key_raises_key_error(interface_with_non_exe) -> None:
    with pytest.raises(KeyError):
        interface_with_non_exe._get_processor("unknown")
