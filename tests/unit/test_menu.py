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


def test_show_menu_returns_to_parent_when_choice_zero_with_menu_stack(
    monkeypatch,
) -> None:
    menu = MenuSystem()
    main_menu = {"title": "main", "options": [("noop", lambda: None)]}
    submenu = {"title": "sub", "options": [("noop", lambda: None)]}
    menu.set_main_menu(main_menu)

    rendered_titles = []
    choices = iter([0, 0])

    monkeypatch.setattr(
        menu,
        "_display_current_menu",
        lambda: rendered_titles.append(menu.current_menu["title"]),
    )
    monkeypatch.setattr(menu, "_get_user_choice", lambda: next(choices))
    monkeypatch.setattr(menu, "_pause", lambda: None)

    with pytest.raises(SystemExit) as exc_info:
        menu.show_menu(submenu)

    assert exc_info.value.code == 0
    assert rendered_titles[:2] == ["sub", "main"]


def test_show_menu_exits_when_choice_zero_on_main_menu(monkeypatch) -> None:
    menu = MenuSystem()
    menu.set_main_menu({"title": "main", "options": [("noop", lambda: None)]})

    monkeypatch.setattr(menu, "_display_current_menu", lambda: None)
    monkeypatch.setattr(menu, "_get_user_choice", lambda: 0)
    monkeypatch.setattr(menu, "_pause", lambda: None)

    with pytest.raises(SystemExit) as exc_info:
        menu.show_menu()

    assert exc_info.value.code == 0
