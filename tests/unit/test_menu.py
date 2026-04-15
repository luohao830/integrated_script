import builtins

import pytest

from integrated_script.config.exceptions import UserInterruptError
from integrated_script.ui.menu import MenuSystem


def test_run_raises_when_main_menu_not_set() -> None:
    menu = MenuSystem()

    with pytest.raises(ValueError):
        menu.run()


def test_get_user_choice_retries_until_valid(monkeypatch) -> None:
    menu = MenuSystem()
    menu.current_menu = {"title": "test", "options": [("opt", lambda: None)]}

    inputs = iter(["", "abc", "1"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(inputs))

    choice = menu._get_user_choice()

    assert choice == 1


def test_get_user_choice_raises_user_interrupt_on_eof(monkeypatch) -> None:
    menu = MenuSystem()
    menu.current_menu = {"title": "test", "options": []}

    def _raise_eof(_prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr(builtins, "input", _raise_eof)

    with pytest.raises(UserInterruptError):
        menu._get_user_choice()
