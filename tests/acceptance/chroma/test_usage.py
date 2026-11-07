import json

from conftest import run, run_text

TINY = ["inspect", "chroma", "--collection", "tiny.l2"]


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
    assert check.stdout == "Checking the target at http://localhost:8099\n"
    assert check.stderr.splitlines()[-2:] == [
        "Error: Weaviate refused the request: the API key is missing or wrong.",
        "Hint: Check WEAVIATE_API_KEY in the environment or in .env.",
    ]
