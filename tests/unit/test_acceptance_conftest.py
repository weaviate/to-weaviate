"""The target rules of the acceptance suite, which live in tests/e2e.py.

From tests/unit, the module name `conftest` is the conftest of this
directory, so these tests import e2e, which tests/acceptance/conftest.py
re-exports.
"""

import e2e
import pytest

CLOUD_URL = "https://abc.c0.europe-west3.gcp.weaviate.cloud"
MADE_UP_KEY = "made-up-cloud-key-t22"
LOCAL_CHROMA = "http://localhost:8001"
LOCAL = {
    "CHROMA_URL": LOCAL_CHROMA,
    "WEAVIATE_URL": "http://localhost:8099",
    "WEAVIATE_GRPC_URL": "http://localhost:50099",
    "WEAVIATE_API_KEY": "admin-key",
}
CLOUD = {"WEAVIATE_URL": CLOUD_URL, "WEAVIATE_API_KEY": MADE_UP_KEY}
VARIABLES = (
    "ACCEPTANCE_TARGET",
    "ACCEPTANCE_VECTORIZER",
    "ACCEPTANCE_NODES",
    "CHROMA_URL",
    "WEAVIATE_URL",
    "WEAVIATE_GRPC_URL",
    "WEAVIATE_API_KEY",
)


@pytest.fixture
def env(monkeypatch):
    """Sets the given variables after it removes every variable of the rules."""
    for name in VARIABLES:
        monkeypatch.delenv(name, raising=False)

    def set_env(values):
        for name, value in values.items():
            monkeypatch.setenv(name, value)

    return set_env


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, "local"), ("", "local"), ("local", "local"), (" Cloud ", "cloud")],
)
def test_target_is_local_unless_the_variable_names_another(env, value, expected):
    if value is not None:
        env({"ACCEPTANCE_TARGET": value})
    assert e2e.target() == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "text2vec-transformers"),
        ("", "text2vec-transformers"),
        ("text2vec-weaviate", "text2vec-weaviate"),
    ],
)
def test_vectorizer_is_the_local_module_unless_the_variable_names_another(
    env, value, expected
):
    if value is not None:
        env({"ACCEPTANCE_VECTORIZER": value})
    assert e2e.vectorizer() == expected


@pytest.mark.parametrize(("value", "expected"), [(None, 1), ("", 1), ("3", 3)])
def test_nodes_is_one_unless_the_variable_names_a_count(env, value, expected):
    if value is not None:
        env({"ACCEPTANCE_NODES": value})
    assert e2e.nodes() == expected


@pytest.mark.parametrize("value", ["0", "three", "-1"])
def test_nodes_refuses_a_value_that_is_not_a_count(env, value):
    env({"ACCEPTANCE_NODES": value})
    with pytest.raises(ValueError) as error:
        e2e.nodes()
    assert str(error.value) == "ACCEPTANCE_NODES is not a whole number of at least 1"


@pytest.mark.parametrize("target", [{}, {"ACCEPTANCE_TARGET": "local"}])
def test_the_local_target_accepts_the_four_test_endpoints(env, target):
    env({**LOCAL, **target})
    assert e2e.endpoint_problems() == []


@pytest.mark.parametrize("target", [{}, {"ACCEPTANCE_TARGET": "local"}])
def test_the_local_target_refuses_a_cloud_cluster(env, target):
    env({**CLOUD, "CHROMA_URL": LOCAL_CHROMA, **target})
    assert e2e.endpoint_problems() == [
        "WEAVIATE_URL is not http://localhost:8099",
        "WEAVIATE_GRPC_URL is not set",
        "WEAVIATE_API_KEY is not the test key",
    ]


def test_an_unknown_target_is_a_problem(env):
    env({**LOCAL, "ACCEPTANCE_TARGET": "staging"})
    assert e2e.endpoint_problems() == ["ACCEPTANCE_TARGET is neither local nor cloud"]


CLOUD_ACCEPTED = {
    "weaviate.cloud host": {},
    "weaviate.network host": {"WEAVIATE_URL": "https://abc.weaviate.network"},
    "any key": {"WEAVIATE_API_KEY": "k"},
    # The tool reads an empty value as unset.
    "empty grpc url": {"WEAVIATE_GRPC_URL": ""},
}


@pytest.mark.parametrize("change", CLOUD_ACCEPTED.values(), ids=CLOUD_ACCEPTED)
def test_the_cloud_target_accepts_a_weaviate_cloud_cluster_and_the_local_chroma(
    env, change
):
    values = {"ACCEPTANCE_TARGET": "cloud", "CHROMA_URL": LOCAL_CHROMA, **CLOUD}
    env({**values, **change})
    assert e2e.endpoint_problems() == []


# (variables that differ from an accepted cloud run, None to unset one; problem)
CLOUD_REFUSED = {
    "no url": ({"WEAVIATE_URL": None}, "WEAVIATE_URL is not set"),
    "http": (
        {"WEAVIATE_URL": CLOUD_URL.replace("https://", "http://")},
        "WEAVIATE_URL does not use https (the cloud target needs TLS)",
    ),
    "other host": (
        {"WEAVIATE_URL": "https://abc.example.com"},
        "the host of WEAVIATE_URL does not end with .weaviate.cloud or "
        ".weaviate.network",
    ),
    "suffix inside the host": (
        {"WEAVIATE_URL": "https://abc.weaviate.cloud.example.com"},
        "the host of WEAVIATE_URL does not end with .weaviate.cloud or "
        ".weaviate.network",
    ),
    "no key": ({"WEAVIATE_API_KEY": None}, "WEAVIATE_API_KEY is not set"),
    "grpc url": (
        {"WEAVIATE_GRPC_URL": "https://grpc-abc.weaviate.cloud"},
        "WEAVIATE_GRPC_URL is set (the cloud target connects without it)",
    ),
    "no chroma url": ({"CHROMA_URL": None}, "CHROMA_URL is not set"),
    "other chroma": (
        {"CHROMA_URL": "https://api.trychroma.com"},
        "CHROMA_URL is not http://localhost:8001 (the source stays the local "
        "test Chroma)",
    ),
}


@pytest.mark.parametrize(
    ("change", "problem"), CLOUD_REFUSED.values(), ids=CLOUD_REFUSED
)
def test_the_cloud_target_names_the_rule_that_a_variable_breaks(env, change, problem):
    values = {"ACCEPTANCE_TARGET": "cloud", "CHROMA_URL": LOCAL_CHROMA, **CLOUD}
    values.update(change)
    env({name: value for name, value in values.items() if value is not None})
    problems = e2e.endpoint_problems()
    assert problems == [problem]
    shown = [v for v in values.values() if v and v not in ("cloud", LOCAL_CHROMA)]
    assert not any(value in problems[0] for value in shown), problems


OWNED = ["Docs_cosine", "Tiny_l2"]


@pytest.mark.parametrize("listed", [[], ["Tiny_l2"], OWNED])
def test_a_target_that_holds_only_suite_collections_is_dedicated(listed):
    assert e2e.dedicated_cluster_problem(listed, OWNED) is None


def test_a_target_with_other_collections_is_not_dedicated():
    listed = ["Docs_cosine", "Zeta", "Alpha"]
    assert e2e.dedicated_cluster_problem(listed, OWNED) == (
        "the target holds collections the suite does not own: Alpha, Zeta; "
        "use a dedicated cluster"
    )
