from conftest import run, weaviate_config, weaviate_count


def test_end_to_end(seeded, tmp_path):
    names = ["--collection", "docs-cosine", "--collection", "tiny.l2"]
    r = run(["inspect", "chroma", *names], tmp_path)
    assert r.exit_code == 0, r
    for cmd in (["check"], ["copy"], ["verify"]):
        r = run(cmd, tmp_path)
        assert r.exit_code == 0 and r.data["status"] == "success", r
    assert weaviate_count("Docs_cosine") == 700 and weaviate_count("Tiny_l2") == 10
    r = run(["copy"], tmp_path)
    assert r.exit_code == 0 and weaviate_count("Docs_cosine") == 700, r


def test_wrong_or_missing_api_key_exits_3(seeded, tmp_path):
    r = run(["inspect", "chroma", "--collection", "tiny.l2"], tmp_path)
    assert r.exit_code == 0, r
    # An empty value counts as unset: run cannot remove a variable.
    for key in ("wrong", ""):
        r = run(["check"], tmp_path, env={"WEAVIATE_API_KEY": key})
        assert r.exit_code == 3 and r.data["code"] == "auth_failed", r
        assert "WEAVIATE_API_KEY" in r.data["hint"]


def test_rq_bits_reach_the_server(seeded, tmp_path):
    r = run(
        [
            "inspect",
            "chroma",
            "--collection",
            "docs-rq",
            "--index",
            "hnsw",
            "--quantizer",
            "rq",
            "--quantizer-bits",
            "8",
        ],
        tmp_path,
    )
    assert r.exit_code == 0, r
    for cmd in (["check"], ["copy"], ["verify"]):
        r = run(cmd, tmp_path)
        assert r.exit_code == 0, r
    vector = weaviate_config("Docs_rq").vector_config["default"]
    assert vector.vector_index_config.quantizer.bits == 8
