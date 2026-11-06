import json
from pathlib import Path

import pytest

from to_weaviate.events import Events


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture(autouse=True)
def _close_events(monkeypatch):
    opened: list[Events] = []
    original = Events.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        opened.append(self)

    monkeypatch.setattr(Events, "__init__", init)
    yield
    for ev in opened:
        ev.close()


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_progress_events_are_throttled(tmp_path):
    clock = FakeClock()
    ev = Events(tmp_path / "events.jsonl", clock)
    for i in range(1000):
        ev.progress("c1", i, 1000)
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 1


def test_progress_writes_again_after_five_seconds(tmp_path):
    clock = FakeClock()
    ev = Events(tmp_path / "events.jsonl", clock)
    ev.progress("c1", 1, 10)
    clock.now = 4.9
    ev.progress("c1", 2, 10)
    clock.now = 5.0
    ev.progress("c1", 3, 10)
    assert [e["copied"] for e in read(tmp_path / "events.jsonl")] == [1, 3]


def test_progress_throttle_is_per_ref(tmp_path):
    ev = Events(tmp_path / "events.jsonl", FakeClock())
    ev.progress("c1", 1, 10)
    ev.progress("c2", 1, 10)
    assert [e["ref"] for e in read(tmp_path / "events.jsonl")] == ["c1", "c2"]


def test_emit_line_shape_and_parent_dir(tmp_path):
    path = tmp_path / "nested" / "events.jsonl"
    ev = Events(path)
    ev.emit("info", "started", ref="c1", count=3)
    [line] = read(path)
    assert line["level"] == "info"
    assert line["event"] == "started"
    assert line["ref"] == "c1"
    assert line["count"] == 3
    assert line["ts"].endswith("+00:00")


def test_progress_line_fields(tmp_path):
    ev = Events(tmp_path / "events.jsonl", FakeClock())
    ev.progress("c1", 5, 10)
    [line] = read(tmp_path / "events.jsonl")
    assert (line["event"], line["ref"], line["copied"], line["total"]) == (
        "progress",
        "c1",
        5,
        10,
    )


def test_events_print_nothing(tmp_path, capsys):
    ev = Events(tmp_path / "events.jsonl", FakeClock())
    ev.emit("info", "x")
    ev.progress("c1", 1, 2)
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
