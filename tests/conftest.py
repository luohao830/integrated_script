import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:
    __import__("numpy")
except ImportError:
    numpy_stub = types.ModuleType("numpy")
    numpy_stub.uint8 = object()
    numpy_stub.frombuffer = lambda raw, dtype=None: raw
    sys.modules["numpy"] = numpy_stub


def pytest_collection_modifyitems(items):
    for item in items:
        item_path = Path(str(item.fspath))
        parts = item_path.parts

        if "tests" in parts and "unit" in parts:
            item.add_marker(pytest.mark.unit)
        elif "tests" in parts and "integration" in parts:
            item.add_marker(pytest.mark.integration)


@pytest.fixture
def sample_config_data() -> dict:
    return {
        "version": "1.0.0",
        "paths": {"input_dir": "", "output_dir": ""},
        "processing": {
            "batch_size": 16,
            "max_workers": 2,
            "timeout": 30,
            "retry_count": 1,
        },
        "ui": {"language": "zh_CN", "theme": "default", "show_progress": True},
        "yolo": {
            "image_formats": [".jpg", ".png"],
            "label_format": ".txt",
            "classes_file": "classes.txt",
            "validate_on_load": True,
        },
    }
