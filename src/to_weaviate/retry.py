import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

MAX_DELAY_SECONDS = 30.0
ATTEMPTS = 7


def with_retry(
    call: Callable[[], T],
    transient: Callable[[Exception], float | None],
    sleep: Callable[[float], None] = time.sleep,
    attempts: int = ATTEMPTS,
) -> T:
    """Run `call`, retrying errors that `transient` accepts.

    `transient(exc)` returns None for an error that must not be retried, else
    the minimum delay in seconds. Every delay is capped at 30 seconds. The last
    exception is re-raised when the attempts are used up.

    There are 7 attempts and 6 delays. Without a `minimum`, a full retry waits
    between 15.75 s and 31.5 s, so a restart of a source or a target that
    lasts a few seconds does not use up the attempts (see
    test_chroma_restart.py).
    A `minimum` from a Retry-After header can raise each delay up to 30 s.
    """
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:
            minimum = transient(exc)
            if minimum is None or attempt == attempts - 1:
                raise
            sleep(backoff_delay(attempt, minimum))
    raise AssertionError("attempts must be at least 1")


def backoff_delay(attempt: int, minimum: float = 0.0) -> float:
    """The delay after failed attempt number `attempt`, counted from 0.

    Random between half of `backoff` and `backoff`, where `backoff` is
    0.5 * 2**attempt seconds capped at MAX_DELAY_SECONDS; at least `minimum`
    and at most MAX_DELAY_SECONDS. The floor of half the backoff keeps a run of
    short delays from giving up inside an outage of a few seconds (see
    test_chroma_restart.py).
    """
    backoff: float = min(MAX_DELAY_SECONDS, 0.5 * 2**attempt)
    return min(
        MAX_DELAY_SECONDS, max(minimum, backoff / 2 + random.uniform(0, backoff / 2))
    )
