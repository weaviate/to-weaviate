import subprocess
import sys

import pytest

from to_weaviate.lock import Locked, exclusive_lock

HOLDER = """
import sys
import time
from pathlib import Path

from to_weaviate.lock import exclusive_lock

with exclusive_lock(Path(sys.argv[1])):
    print("held", flush=True)
    time.sleep(60)
"""

WITHOUT_FCNTL = """
import sys
from pathlib import Path

sys.modules["fcntl"] = None  # import fcntl now raises ImportError
from to_weaviate.lock import exclusive_lock

path = Path(sys.argv[1])
with exclusive_lock(path), exclusive_lock(path):
    print("ran")
"""


def test_a_second_lock_is_refused_until_the_first_ends(tmp_path):
    pytest.importorskip("fcntl")
    path = tmp_path / "copy.lock"
    with exclusive_lock(path):
        with pytest.raises(Locked), exclusive_lock(path):
            pass
    with exclusive_lock(path):
        pass
    assert path.exists()


def test_an_existing_unlocked_file_does_not_block(tmp_path):
    path = tmp_path / "copy.lock"
    path.write_text("")
    with exclusive_lock(path):
        pass


def test_a_killed_holder_leaves_no_lock(tmp_path):
    pytest.importorskip("fcntl")
    path = tmp_path / "copy.lock"
    with subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(path)], stdout=subprocess.PIPE, text=True
    ) as holder:
        try:
            assert holder.stdout is not None
            assert holder.stdout.readline() == "held\n"
            with pytest.raises(Locked), exclusive_lock(path):
                pass
        finally:
            holder.kill()
    with exclusive_lock(path):
        pass


def test_without_fcntl_the_module_imports_and_runs_without_a_lock(tmp_path):
    done = subprocess.run(
        [sys.executable, "-c", WITHOUT_FCNTL, str(tmp_path / "copy.lock")],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert (done.returncode, done.stdout) == (0, "ran\n"), done.stderr
