import json
import os

from chroma.seed import (
    add_to_tiny_l2,
    delete_from_tiny_l2,
    extra_tiny_l2_ids,
    tiny_l2_count,
)
from conftest import read_plan, run, run_text, weaviate_by_source_id, write_plan

from to_weaviate.settings import without_userinfo

TINY = ["inspect", "chroma", "--collection", "tiny.l2"]
NOTE = {
    "source_key": "note",
    "name": "note",
    "type": "text",
    "tokenization": "word",
    "searchable": True,
}


def test_usage_errors_exit_2_and_name_the_value(seeded, tmp_path):
    first = tmp_path / "first"
    first.mkdir()
    for args, message in [
        (
            ["inspect", "chroma", "--database", "nope"],
            "Chroma tenant default_tenant has no collection in database nope.",
        ),
        (
            ["inspect", "chroma", "--collection", "nope"],
            "Chroma has no collection named nope.",
        ),
    ]:
        r = run(args, first)
        assert r.exit_code == 2 and r.data["code"] == "usage", r
        assert r.data["message"] == message
    assert list(first.iterdir()) == []

    r = run(TINY, tmp_path)
    assert r.exit_code == 0, r
    for args, message in [
        (TINY, f"{tmp_path}/plan.yaml exists."),
        (["copy", "--collection", "nope"], "--collection nope matches no plan entry."),
        (["copy", "--parallelism", "9"], "--parallelism 9 is outside 1 to 8."),
    ]:
        r = run(args, tmp_path)
        assert r.exit_code == 2 and r.data["code"] == "usage", r
        assert r.data["message"] == message
    assert not (tmp_path / "state.sqlite").exists()


def test_text_mode_prints_lines_and_the_error_on_stderr(seeded, tmp_path):
    r = run(TINY, tmp_path)
    assert r.exit_code == 0, r
    status = run_text(["status"], tmp_path)
    assert status.returncode == 0, status
    lines = status.stdout.splitlines()
    assert lines[0].startswith("tiny.l2 -> Tiny_l2: copied 0 of 10, rejects 0")
    assert lines[-1] == "Total: 10 records"
    try:
        json.loads(status.stdout)
    except ValueError:
        pass
    else:
        raise AssertionError("status printed JSON in text mode")
    check = run_text(["check"], tmp_path, env={"WEAVIATE_API_KEY": "made-up-key-b9"})
    assert check.returncode == 3, check
    # In text mode the progress line goes to stdout and the error to stderr.
    # The CLI prints the stripped URL without its userinfo.
    shown = without_userinfo(os.environ["WEAVIATE_URL"].strip())
    assert check.stdout == f"Checking the target at {shown}\n"
    assert check.stderr.splitlines()[-2:] == [
        "Error: Weaviate refused the request: the API key is missing or wrong.",
        "Hint: Check WEAVIATE_API_KEY in the environment or in .env.",
    ]


def test_a_rejected_record_is_written_by_sync_once_the_plan_is_fixed(seeded, tmp_path):
    r = run(TINY, tmp_path)
    assert r.exit_code == 0, r
    added = extra_tiny_l2_ids(10, 1)
    try:
        # A record added after the inspection holds a key that plan.yaml lacks.
        add_to_tiny_l2(10, 1, {"note": "added later"})
        r = run(["copy"], tmp_path)
        assert r.exit_code == 1 and r.data["code"] == "rejects", r
        assert "then to-weaviate sync writes the record" in r.data["hint"], r
        plan = read_plan(tmp_path)
        plan["collections"][0]["target"]["properties"].append(NOTE)
        write_plan(tmp_path, plan)
        # The run adds the property and reads no page again.
        r = run(["copy"], tmp_path)
        assert r.exit_code == 1 and r.data["collections"][0]["rejects"] == 1, r
        assert r.data["added_properties"] == [
            {"collection": "Tiny_l2", "property": "note"}
        ], r
        # The path that the hint gives. The plan entry changed, so every
        # record is written again.
        r = run(["sync"], tmp_path)
        assert r.exit_code == 0 and r.data["status"] == "success", r
        row = r.data["collections"][0]
        assert (row["copied"], row["rejects"], row["plan_changed"]) == (11, 0, True), r
        r = run(["verify"], tmp_path)
        assert r.exit_code == 0 and r.data["status"] == "success", r
        found = weaviate_by_source_id("Tiny_l2", added[0])
        assert found is not None and found.properties["note"] == "added later"
    finally:
        delete_from_tiny_l2(added)
        assert tiny_l2_count() == 10
