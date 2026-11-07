import json

import pytest
from chroma.seed import count
from conftest import run, weaviate_by_source_id, weaviate_count

TARGET = "Rejects"
# (inspect options, rejected source id, reason). Chroma accepts both records:
# a float under a key that holds ints elsewhere, and a record without a
# document.
REJECTS = {
    "type-after-the-sampled-page": (
        ["--sample-pages", "1"],
        "rej-300",
        "rank: cannot convert number value 300.5 to int without losing data",
    ),
    "no-document-to-embed": (
        ["--vectors", "embed", "--vectorizer", "text2vec-transformers"],
        "rej-005",
        "no document to embed",
    ),
}


@pytest.mark.parametrize(("args", "source_id", "reason"), REJECTS.values(), ids=REJECTS)
def test_a_record_that_cannot_be_written_is_a_reject(
    seeded, tmp_path, args, source_id, reason
):
    r = run(["inspect", "chroma", "--collection", "rejects", *args], tmp_path)
    assert r.exit_code == 0, r
    r = run(["check"], tmp_path)
    assert r.exit_code == 0, r
    total = count("rejects")
    r = run(["copy"], tmp_path)
    assert r.exit_code == 1 and r.data["code"] == "rejects", r
    row = r.data["collections"][0]
    assert (row["copied"], row["rejects"], row["total"]) == (total - 1, 1, total), r
    lines = (tmp_path / "rejects.jsonl").read_text(encoding="utf-8").splitlines()
    found = [json.loads(line) for line in lines]
    assert [(f["source_id"], f["reason"]) for f in found] == [(source_id, reason)]
    assert weaviate_count(TARGET) == total - 1
    assert weaviate_by_source_id(TARGET, source_id) is None
    r = run(["verify"], tmp_path)
    assert r.exit_code == 1 and r.data["code"] == "verification_failed", r
    checks = {c["name"]: c for c in r.data["checks"]}
    assert not checks["count"]["passed"]
    assert checks["ids"]["detail"] == f'missing 1, extra 0; missing ids ["{source_id}"]'
