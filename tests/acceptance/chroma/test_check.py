import pytest
from chroma.seed import (
    ADD_BATCH,
    OPENAI_ADA,
    OPENAI_ADA_FUNCTION,
    delete_collection,
    make_collection,
    order_records,
)
from conftest import (
    enabled_vectorizer,
    read_plan,
    run,
    weaviate_config,
    weaviate_count,
    weaviate_modules,
)

# Neither variable reaches the command from the shell.
NO_OPENAI_KEY = {"OPENAI_APIKEY": "", "OPENAI_API_KEY": ""}
MADE_UP_OPENAI_KEY = "made-up-openai-key-b9"
# Weaviate 1.40.4 enables every API-based module by default, text2vec-openai
# and text2vec-weaviate among them. The test Weaviate does not enable
# text2vec-ollama; Weaviate Cloud does and leaves text2vec-transformers disabled.
CANDIDATE_MODULES = (
    "text2vec-ollama",
    "text2vec-transformers",
    "text2vec-contextionary",
    "text2vec-gpt4all",
    "text2vec-model2vec",
)


def disabled_module() -> str:
    """The first of CANDIDATE_MODULES that /v1/meta of the target does not list."""
    enabled = weaviate_modules()
    for module in CANDIDATE_MODULES:
        if module not in enabled:
            return module
    pytest.fail(
        f"the target enables every module of CANDIDATE_MODULES: "
        f"{sorted(CANDIDATE_MODULES)}; this case needs a module that it does "
        "not enable"
    )


def test_check_needs_the_provider_key_and_an_enabled_module(seeded, tmp_path):
    inspect = ["inspect", "chroma", "--collection", "tiny.l2"]
    r = run([*inspect, "--vectorizer", "text2vec-openai"], tmp_path)
    assert r.exit_code == 0, r
    r = run(["check"], tmp_path, env=NO_OPENAI_KEY)
    assert r.exit_code == 3 and r.data["code"] == "missing_credential", r
    assert r.data["message"] == "OPENAI_API_KEY is not set"
    # run fails the test when the key appears in the output.
    key = {**NO_OPENAI_KEY, "OPENAI_API_KEY": MADE_UP_OPENAI_KEY}
    r = run(["check"], tmp_path, env=key)
    assert r.exit_code == 0, r

    disabled = disabled_module()
    embedded = tmp_path / "embed"
    embedded.mkdir()
    r = run([*inspect, "--vectors", "embed", "--vectorizer", disabled], embedded)
    assert r.exit_code == 0, r
    r = run(["check"], embedded)
    assert r.exit_code == 2 and r.data["code"] == "invalid_plan", r
    assert [e["code"] for e in r.data["errors"]] == ["module_not_enabled"], r
    message = r.data["errors"][0]["message"]
    assert message.startswith(
        f"module {disabled} is not enabled on the target"
    ), message
    assert enabled_vectorizer() in message.partition("enabled modules: ")[2]


def test_openai_ada_002_plan_is_accepted_by_the_target(tmp_path):
    """The drafted vectorizer of an ada-002 collection passes check and copy.

    Weaviate validates the model name when it creates the collection. A copy
    that keeps the vectors sends no request to OpenAI, so a made-up key serves.
    """
    key = {**NO_OPENAI_KEY, "OPENAI_API_KEY": MADE_UP_OPENAI_KEY}
    records = order_records(0, 10)
    try:
        make_collection(OPENAI_ADA, records, ADD_BATCH, OPENAI_ADA_FUNCTION)
        r = run(["inspect", "chroma", "--collection", OPENAI_ADA], tmp_path)
        assert r.exit_code == 0, r
        (vector,) = read_plan(tmp_path)["collections"][0]["target"]["vectors"]
        assert vector["vectorizer"]["module"] == "text2vec-openai", vector
        assert vector["vectorizer"]["options"] == {"model": "ada"}, vector
        for command in (["check"], ["copy"], ["verify"]):
            r = run(command, tmp_path, env=key)
            assert r.exit_code == 0 and r.data["status"] == "success", r
        vectorizer = weaviate_config("Openai_ada").vector_config["default"].vectorizer
        assert vectorizer.vectorizer.value == "text2vec-openai"
        assert vectorizer.model["model"] == "ada"
        assert weaviate_count("Openai_ada") == len(records)
    finally:
        delete_collection(OPENAI_ADA)
