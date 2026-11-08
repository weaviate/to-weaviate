from unittest.mock import patch

import pytest

from to_weaviate.retry import with_retry


def test_with_retry_raises_permanent_error_at_once():
    calls = []

    def call():
        calls.append(1)
        raise ValueError("bad")

    sleeps: list[float] = []
    with pytest.raises(ValueError):
        with_retry(call, lambda exc: None, sleep=sleeps.append)
    assert len(calls) == 1 and sleeps == []


def test_with_retry_returns_after_transient_failures():
    results = [RuntimeError("a"), RuntimeError("b"), "ok"]

    def call():
        r = results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    sleeps: list[float] = []
    assert with_retry(call, lambda exc: 0.0, sleep=sleeps.append) == "ok"
    assert len(sleeps) == 2


def test_with_retry_reraises_last_exception_after_all_attempts():
    errors = [RuntimeError(str(i)) for i in range(7)]
    sleeps: list[float] = []

    def call():
        raise errors.pop(0)

    with pytest.raises(RuntimeError, match="6"):
        with_retry(call, lambda exc: 0.0, sleep=sleeps.append)
    assert len(sleeps) == 6


def test_with_retry_delay_is_capped_at_30_seconds():
    sleeps: list[float] = []

    def call():
        raise RuntimeError("x")

    with patch("to_weaviate.retry.random.uniform", side_effect=lambda a, b: b):
        with pytest.raises(RuntimeError):
            with_retry(call, lambda exc: 100.0, sleep=sleeps.append)
    assert sleeps == [30] * 6


def test_with_retry_backoff_grows_and_honors_minimum():
    sleeps: list[float] = []

    def call():
        raise RuntimeError("x")

    with patch("to_weaviate.retry.random.uniform", side_effect=lambda a, b: b):
        with pytest.raises(RuntimeError):
            with_retry(call, lambda exc: 0.0, sleep=sleeps.append)
    assert sleeps == [0.5, 1, 2, 4, 8, 16]

    sleeps.clear()
    with patch("to_weaviate.retry.random.uniform", return_value=0.0):
        with pytest.raises(RuntimeError):
            with_retry(call, lambda exc: 2.0, sleep=sleeps.append)
    assert sleeps == [2.0, 2.0, 2.0, 2.0, 4.0, 8.0]


def test_a_full_retry_waits_at_least_15_seconds_and_at_most_31():
    def call():
        raise RuntimeError("x")

    for pick, expected in (
        (lambda a, b: a, [0.25, 0.5, 1, 2, 4, 8]),
        (lambda a, b: b, [0.5, 1, 2, 4, 8, 16]),
    ):
        sleeps: list[float] = []
        with patch("to_weaviate.retry.random.uniform", side_effect=pick):
            with pytest.raises(RuntimeError):
                with_retry(call, lambda exc: 0.0, sleep=sleeps.append)
        assert sleeps == expected
    assert sum([0.25, 0.5, 1, 2, 4, 8]) == 15.75
    assert sum([0.5, 1, 2, 4, 8, 16]) == 31.5
