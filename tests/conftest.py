import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(autouse=True, scope="session")
def throwaway_home(tmp_path_factory):
    """Every test, and every process a test starts, gets an empty home folder. The app keeps its settings,
    its model files and its log under the home folder, and no test may read or write the real ones."""
    real = os.environ.get("HOME")
    home = tmp_path_factory.mktemp("home")
    os.environ["HOME"] = str(home)
    yield home
    if real is None:
        del os.environ["HOME"]
    else:
        os.environ["HOME"] = real
