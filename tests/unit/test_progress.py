from integrated_script.core.progress import (
    ProgressManager,
    process_with_progress,
    progress_context,
)


class _DummyBar:
    def __init__(self):
        self.updated: list[int] = []
        self.descriptions: list[str] = []
        self.closed = False

    def update(self, n: int) -> None:
        self.updated.append(n)

    def set_description(self, description: str) -> None:
        self.descriptions.append(description)

    def close(self) -> None:
        self.closed = True


def test_create_progress_bar_returns_none_when_disabled() -> None:
    manager = ProgressManager(show_progress=False)

    bar = manager.create_progress_bar(total=10, description="x")

    assert bar is None


def test_create_progress_bar_success_and_state(monkeypatch) -> None:
    manager = ProgressManager(show_progress=True)
    fake_bar = _DummyBar()

    monkeypatch.setattr(
        "integrated_script.core.progress.tqdm",
        lambda **_kwargs: fake_bar,
    )

    bar = manager.create_progress_bar(total=5, description="处理中", unit="item")

    assert bar is fake_bar
    assert manager.progress_bar is fake_bar


def test_create_progress_bar_returns_none_when_tqdm_raises(monkeypatch) -> None:
    manager = ProgressManager(show_progress=True)

    def _raise_tqdm(**_kwargs):
        raise RuntimeError("tqdm-fail")

    monkeypatch.setattr("integrated_script.core.progress.tqdm", _raise_tqdm)

    bar = manager.create_progress_bar(total=5, description="处理中")

    assert bar is None


def test_update_progress_updates_bar_and_description() -> None:
    manager = ProgressManager(show_progress=True)
    bar = _DummyBar()
    manager.progress_bar = bar

    manager.update_progress(2, description="step-2")

    assert manager._processed_items == 2
    assert bar.updated == [2]
    assert bar.descriptions == ["step-2"]


def test_update_progress_swallows_bar_exception() -> None:
    manager = ProgressManager(show_progress=True)

    class _RaiseBar:
        def update(self, _n):
            raise RuntimeError("update-fail")

        def set_description(self, _description):
            return None

    manager.progress_bar = _RaiseBar()

    manager.update_progress(1, description="x")

    assert manager._processed_items == 1


def test_update_progress_prints_when_no_bar(monkeypatch) -> None:
    manager = ProgressManager(show_progress=True)
    manager._total_items = 10
    printed = []

    monkeypatch.setattr(
        "builtins.print",
        lambda *args, **kwargs: printed.append((args, kwargs)),
    )

    manager.update_progress(1)

    assert manager._processed_items == 1
    assert printed


def test_close_progress_bar_closes_and_resets() -> None:
    manager = ProgressManager(show_progress=True)
    bar = _DummyBar()
    manager.progress_bar = bar
    manager._start_time = 1.0

    manager.close_progress_bar()

    assert bar.closed is True
    assert manager.progress_bar is None


def test_close_progress_bar_handles_close_exception() -> None:
    manager = ProgressManager(show_progress=True)

    class _RaiseCloseBar:
        def close(self):
            raise RuntimeError("close-fail")

    manager.progress_bar = _RaiseCloseBar()

    manager.close_progress_bar()

    assert manager.progress_bar is None


def test_close_progress_bar_prints_when_no_bar(monkeypatch) -> None:
    manager = ProgressManager(show_progress=True)
    manager.progress_bar = None
    manager._total_items = 2
    printed = []

    monkeypatch.setattr(
        "builtins.print",
        lambda *args, **kwargs: printed.append((args, kwargs)),
    )

    manager.close_progress_bar()

    assert printed


def test_progress_manager_context_calls_close(monkeypatch) -> None:
    manager = ProgressManager(show_progress=True)
    called = {"closed": False}

    def _fake_close():
        called["closed"] = True

    monkeypatch.setattr(manager, "close_progress_bar", _fake_close)

    with manager as ctx:
        assert ctx is manager

    assert called["closed"] is True


def test_progress_context_creates_and_closes(monkeypatch) -> None:
    called = {"create": False, "close": False}

    def _fake_create(*_args, **_kwargs):
        called["create"] = True
        return None

    def _fake_close(*_args, **_kwargs):
        called["close"] = True

    monkeypatch.setattr(ProgressManager, "create_progress_bar", _fake_create)
    monkeypatch.setattr(ProgressManager, "close_progress_bar", _fake_close)

    with progress_context(3, "desc", show_progress=True) as manager:
        assert isinstance(manager, ProgressManager)

    assert called == {"create": True, "close": True}


def test_process_with_progress_success_path(monkeypatch) -> None:
    class _DummyContext:
        def __init__(self):
            self.manager = ProgressManager(show_progress=False)
            self.calls: list[int] = []

            def _update(n=1, _description=None):
                self.calls.append(n)

            self.manager.update_progress = _update

        def __enter__(self):
            return self.manager

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        "integrated_script.core.progress.progress_context",
        lambda *_args, **_kwargs: _DummyContext(),
    )

    result = process_with_progress([1, 2, 3], lambda item: item * 2)

    assert result == [2, 4, 6]


def test_process_with_progress_records_none_without_error_handler(
    monkeypatch,
) -> None:
    class _DummyContext:
        def __init__(self):
            self.manager = ProgressManager(show_progress=False)
            self.manager.update_progress = lambda *_args, **_kwargs: None

        def __enter__(self):
            return self.manager

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        "integrated_script.core.progress.progress_context",
        lambda *_args, **_kwargs: _DummyContext(),
    )

    def _processor(item):
        if item == 2:
            raise ValueError("bad")
        return item

    result = process_with_progress([1, 2, 3], _processor)

    assert result == [1, None, 3]


def test_process_with_progress_uses_error_handler(monkeypatch) -> None:
    class _DummyContext:
        def __init__(self):
            self.manager = ProgressManager(show_progress=False)
            self.manager.update_progress = lambda *_args, **_kwargs: None

        def __enter__(self):
            return self.manager

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        "integrated_script.core.progress.progress_context",
        lambda *_args, **_kwargs: _DummyContext(),
    )

    def _processor(item):
        if item == 2:
            raise ValueError("bad")
        return item

    def _error_handler(_error, item):
        return f"handled-{item}"

    result = process_with_progress(
        [1, 2, 3],
        _processor,
        error_handler=_error_handler,
    )

    assert result == [1, "handled-2", 3]


def test_process_with_progress_falls_back_to_none_when_error_handler_fails(
    monkeypatch,
) -> None:
    class _DummyContext:
        def __init__(self):
            self.manager = ProgressManager(show_progress=False)
            self.manager.update_progress = lambda *_args, **_kwargs: None

        def __enter__(self):
            return self.manager

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        "integrated_script.core.progress.progress_context",
        lambda *_args, **_kwargs: _DummyContext(),
    )

    def _processor(item):
        if item == 2:
            raise ValueError("bad")
        return item

    def _error_handler(_error, _item):
        raise RuntimeError("handler-fail")

    result = process_with_progress([1, 2], _processor, error_handler=_error_handler)

    assert result == [1, None]
