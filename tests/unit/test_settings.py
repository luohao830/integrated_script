from pathlib import Path

import pytest

from integrated_script.config.exceptions import ConfigurationError
from integrated_script.config.settings import ConfigManager
from integrated_script.version import get_version


def test_creates_default_config_file_when_missing(tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    assert not config_file.exists()

    manager = ConfigManager(config_file=config_file, auto_save=False)

    assert config_file.exists()
    assert manager.get("version") == get_version()
    assert manager.get("paths.temp_dir") == "temp"


def test_set_and_get_nested_value_without_auto_save(tmp_path: Path) -> None:
    manager = ConfigManager(config_file=tmp_path / "config.json", auto_save=False)

    manager.set("paths.input_dir", "/data/input")

    assert manager.get("paths.input_dir") == "/data/input"
    assert manager.get("paths.not_exists", "fallback") == "fallback"


def test_update_merges_nested_config(tmp_path: Path) -> None:
    manager = ConfigManager(config_file=tmp_path / "config.json", auto_save=False)

    manager.update({"processing": {"batch_size": 8}})

    assert manager.get("processing.batch_size") == 8
    assert manager.get("processing.max_workers") == 4


def test_validate_raises_for_invalid_processing_type(tmp_path: Path) -> None:
    manager = ConfigManager(config_file=tmp_path / "config.json", auto_save=False)
    manager.set("processing.batch_size", "bad")

    with pytest.raises(ConfigurationError):
        manager.validate()


def test_load_invalid_json_raises_configuration_error(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.json"
    config_file.write_text("{ invalid json", encoding="utf-8")

    with pytest.raises(ConfigurationError):
        ConfigManager(config_file=config_file, auto_save=False)


def test_get_all_does_not_mutate_internal_nested_config(tmp_path: Path) -> None:
    manager = ConfigManager(config_file=tmp_path / "config.json", auto_save=False)

    all_config = manager.get_all()
    all_config["paths"]["temp_dir"] = "mutated-temp"

    assert manager.get("paths.temp_dir") == "temp"


def test_load_from_file_updates_in_memory_config(tmp_path: Path) -> None:
    source_file = tmp_path / "source.json"
    source_file.write_text('{"processing": {"batch_size": 42}}', encoding="utf-8")

    manager = ConfigManager(config_file=tmp_path / "config.json", auto_save=False)
    manager.load_from_file(source_file)

    assert manager.get("processing.batch_size") == 42
    assert manager.get("processing.max_workers") == 4


def test_save_to_file_writes_current_config_to_target(tmp_path: Path) -> None:
    manager = ConfigManager(config_file=tmp_path / "config.json", auto_save=False)
    manager.set("ui.theme", "dark")

    target_file = tmp_path / "export.json"
    manager.save_to_file(target_file)

    exported = ConfigManager(config_file=target_file, auto_save=False)
    assert exported.get("ui.theme") == "dark"
