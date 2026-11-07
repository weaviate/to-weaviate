import json

from chroma.seed import (
    PROXY_HEADER_TOKEN,
    PROXY_PASSWORD,
    PROXY_QUERY_TOKEN,
    PROXY_USER,
)
from conftest import read_plan, run

SECRETS = (PROXY_USER, PROXY_PASSWORD, PROXY_QUERY_TOKEN, PROXY_HEADER_TOKEN)
# The config keys with a value that the stored form drops; default_headers
# holds the header token.
DROPPED = ("api_key_env_var", "api_version", "default_headers", "deployment_id")


def _assert_no_secret(texts: dict[str, str]) -> None:
    for where, text in texts.items():
        for secret in SECRETS:
            assert secret not in text, f"{secret} is in {where}"


def test_embedding_function_secrets_are_not_stored_or_printed(seeded, tmp_path):
    r = run(["inspect", "chroma", "--collection", "openai-proxy"], tmp_path)
    assert r.exit_code == 0, r
    _assert_no_secret(
        {
            "plan.yaml": (tmp_path / "plan.yaml").read_text(encoding="utf-8"),
            "inspect.json": (tmp_path / "inspect.json").read_text(encoding="utf-8"),
            "stdout": json.dumps(r.data),
            "stderr": r.stderr,
        }
    )
    notes = r.data["collections"][0]["notes"]
    dropped = [n for n in notes if "are not stored" in n]
    assert len(dropped) == 1, notes
    for key in DROPPED:
        assert key in dropped[0], (key, dropped[0])
    entry = read_plan(tmp_path)["collections"][0]
    stored = entry["source"]["embedding"]["config"]
    assert stored["api_base"] == "https://proxy.example.invalid/openai/v1"
    assert "default_headers" not in stored
    assert [v["vectorizer"] for v in entry["target"]["vectors"]] == [None]
    r = run(["check"], tmp_path)
    assert r.exit_code == 0, r
    _assert_no_secret({"check stdout": json.dumps(r.data), "check stderr": r.stderr})
