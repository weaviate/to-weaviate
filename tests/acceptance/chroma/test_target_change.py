from conftest import read_plan, run, weaviate_count, weaviate_exists, write_plan

from to_weaviate.lock import exclusive_lock

TARGET = "Tiny_l2"
RENAMED = "Tiny_l2_renamed"


def _set_target(workdir, name: str) -> None:
    plan = read_plan(workdir)
    plan["collections"][0]["target"]["name"] = name
    write_plan(workdir, plan)


def _copied(workdir) -> int:
    r = run(["status"], workdir)
    assert r.exit_code == 0, r
    return int(r.data["collections"][0]["copied"])


def test_a_changed_target_and_a_held_lock_stop_copy(seeded, tmp_path):
    r = run(["inspect", "chroma", "--collection", "tiny.l2"], tmp_path)
    assert r.exit_code == 0, r
    r = run(["copy"], tmp_path)
    assert r.exit_code == 0, r
    assert _copied(tmp_path) == 10

    _set_target(tmp_path, RENAMED)
    r = run(["copy"], tmp_path)
    assert r.exit_code == 2 and r.data["code"] == "state_mismatch", r
    assert not weaviate_exists(RENAMED)
    assert _copied(tmp_path) == 10

    _set_target(tmp_path, TARGET)
    r = run(["copy"], tmp_path)
    assert r.exit_code == 0, r
    assert _copied(tmp_path) == 10 and weaviate_count(TARGET) == 10

    # The lock belongs to an open file of this process, so the copy process
    # finds it held.
    with exclusive_lock(tmp_path / "copy.lock"):
        r = run(["copy"], tmp_path)
    assert r.exit_code == 2 and r.data["code"] == "usage", r
    assert r.data["message"] == f"Another to-weaviate copy uses {tmp_path}."
    assert _copied(tmp_path) == 10
