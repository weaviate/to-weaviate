import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

MAX_DELAY_SECONDS = 30.0


def with_retry(
    call: Callable[[], T],
    transient: Callable[[Exception], float | None],
    sleep: Callable[[float], None] = time.sleep,
    attempts: int = 6,
) -> T:
    """Run `call`, retrying errors that `transient` accepts.

    `transient(exc)` returns None for an error that must not be retried, else
    the minimum delay in seconds. Every delay is capped at 30 seconds. The last
    exception is re-raised when the attempts are used up.
    """
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:
            minimum = transient(exc)
            if minimum is None or attempt == attempts - 1:
                raise
            backoff = min(MAX_DELAY_SECONDS, 0.5 * 2**attempt)
            sleep(min(MAX_DELAY_SECONDS, max(minimum, random.uniform(0, backoff))))
    raise AssertionError("attempts must be at least 1")
