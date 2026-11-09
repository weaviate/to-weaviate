from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl
except ImportError:  # Windows has no fcntl.
    fcntl = None  # type: ignore[assignment]


class Locked(Exception):
    """Another open file of the path holds the lock."""


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive `flock` on `path` for the block, without waiting.

    Raises Locked when another open file of `path` holds the lock, also in
    the same process. The lock belongs to the open file, so the operating
    system releases it when the file is closed or the process ends: a killed
    run leaves no lock. `path` is created when missing and never removed.
    Without `fcntl` the block runs without a lock.
    """
    if fcntl is None:
        yield
        return
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Locked(str(path)) from exc
        yield
