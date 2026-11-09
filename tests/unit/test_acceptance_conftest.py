"""The target rules of the acceptance suite, which live in tests/e2e.py.

From tests/unit, the module name `conftest` is the conftest of this
directory, so these tests import e2e, which tests/acceptance/conftest.py
re-exports.
"""

import e2e
import pytest

pytest_plugins = ["pytester"]

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
    "ACCEPTANCE_CHROMA",
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


class FakeServer:
    """Stands for the client: cluster.nodes() lists `count` nodes and counts calls."""

    def __init__(self, count):
        self.calls = 0
        self.cluster = self
        self._count = count

    def nodes(self):
        self.calls += 1
        return [object()] * self._count


@pytest.fixture
def server(monkeypatch):
    """Returns a function that installs a fake server of the given node count."""
    monkeypatch.setattr(e2e, "_SERVER_NODES", None)

    def install(count):
        fake = FakeServer(count)
        monkeypatch.setattr(e2e, "_weaviate", lambda: fake)
        return fake

    return install


@pytest.mark.parametrize("value", [None, ""])
def test_nodes_asks_the_server_when_the_variable_is_unset(env, server, value):
    if value is not None:
        env({"ACCEPTANCE_NODES": value})
    fake = server(3)
    assert e2e.nodes() == 3
    assert e2e.nodes() == 3
    assert fake.calls == 1


def test_nodes_is_the_variable_when_it_is_set(env, server):
    env({"ACCEPTANCE_NODES": "3"})
    fake = server(1)
    assert e2e.nodes() == 3
    assert fake.calls == 0


@pytest.mark.parametrize("value", ["0", "three"])
def test_a_malformed_node_count_is_a_problem(env, server, value):
    env(LOCAL)
    env({"ACCEPTANCE_NODES": value})
    fake = server(3)
    assert e2e.endpoint_problems() == [
        "ACCEPTANCE_NODES is not a whole number of at least 1"
    ]
    assert fake.calls == 0


@pytest.mark.parametrize(
    ("variable", "count", "problems"),
    [
        ("3", 1, ["ACCEPTANCE_NODES is 3 but the cluster has 1 node"]),
        ("1", 3, ["ACCEPTANCE_NODES is 1 but the cluster has 3 nodes"]),
        ("3", 3, []),
        (None, 3, []),
    ],
)
@pytest.mark.parametrize("target", ["local", "cloud"])
def test_a_node_count_that_differs_from_the_server_is_a_problem(
    env, server, target, variable, count, problems
):
    env(LOCAL if target == "local" else {**CLOUD, "CHROMA_URL": LOCAL_CHROMA})
    env({"ACCEPTANCE_TARGET": target})
    if variable is not None:
        env({"ACCEPTANCE_NODES": variable})
    server(count)
    assert e2e.endpoint_problems() == problems


@pytest.mark.parametrize("value", ["0", "three", "-1"])
def test_nodes_refuses_a_value_that_is_not_a_count(env, value):
    env({"ACCEPTANCE_NODES": value})
    with pytest.raises(ValueError) as error:
        e2e.nodes()
    assert str(error.value) == "ACCEPTANCE_NODES is not a whole number of at least 1"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "single"),
        ("", "single"),
        ("single", "single"),
        ("distributed", "distributed"),
        (" Distributed ", "distributed"),
    ],
)
def test_chroma_kind_is_single_unless_the_variable_names_distributed(
    env, value, expected
):
    if value is not None:
        env({"ACCEPTANCE_CHROMA": value})
    assert e2e.chroma_kind() == expected


@pytest.mark.parametrize("value", ["cloud", "local", "multi"])
def test_chroma_kind_refuses_another_value(env, value):
    env({"ACCEPTANCE_CHROMA": value})
    with pytest.raises(ValueError) as error:
        e2e.chroma_kind()
    assert str(error.value) == "ACCEPTANCE_CHROMA is neither single nor distributed"


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
    "suffix inside a longer host": (
        {"WEAVIATE_URL": "https://abc.weaviate.cloud.evil.example"},
        "the host of WEAVIATE_URL does not end with .weaviate.cloud or "
        ".weaviate.network",
    ),
    "suffix in the userinfo": (
        {"WEAVIATE_URL": "https://abc.weaviate.cloud@evil.example"},
        "the host of WEAVIATE_URL does not end with .weaviate.cloud or "
        ".weaviate.network",
    ),
    "suffix in the query": (
        {"WEAVIATE_URL": "https://evil.example/?h=.weaviate.cloud"},
        "the host of WEAVIATE_URL does not end with .weaviate.cloud or "
        ".weaviate.network",
    ),
    "suffix in the fragment": (
        {"WEAVIATE_URL": "https://evil.example#.weaviate.cloud"},
        "the host of WEAVIATE_URL does not end with .weaviate.cloud or "
        ".weaviate.network",
    ),
    "not a url": ({"WEAVIATE_URL": "https://[abc"}, "WEAVIATE_URL is not a URL"),
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


@pytest.mark.parametrize(
    "listed", [{"text2vec-transformers"}, {"a", "text2vec-transformers"}]
)
def test_enabled_vectorizer_is_the_vectorizer_that_the_target_lists(
    env, monkeypatch, listed
):
    monkeypatch.setattr(e2e, "weaviate_modules", lambda: listed)
    assert e2e.enabled_vectorizer() == "text2vec-transformers"


def test_a_vectorizer_that_the_target_does_not_list_fails_the_test(env, monkeypatch):
    env({"ACCEPTANCE_VECTORIZER": "text2vec-weaviate"})
    monkeypatch.setattr(e2e, "weaviate_modules", lambda: {"text2vec-transformers"})
    with pytest.raises(pytest.fail.Exception) as failure:
        e2e.enabled_vectorizer()
    assert str(failure.value) == (
        "/v1/meta of the target does not list text2vec-weaviate; set "
        "ACCEPTANCE_VECTORIZER to a module that the target enables"
    )


# A conftest and a test module like those of tests/acceptance: the hook runs
# the guard, and the infra test requests a module-scoped fixture that stands
# for one that runs docker or kubectl.
GUARD_CONFTEST = """
import e2e
import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    e2e.fail_infra_on_cloud(item)
"""
INFRA_TEST = """
import pytest


@pytest.fixture(scope="module")
def server():
    open("fixture-ran", "w").close()


@pytest.mark.infra
def test_restarts_a_server(server):
    pass
"""


@pytest.mark.parametrize(("target", "ran"), [("cloud", False), ("local", True)])
def test_the_infra_guard_runs_before_the_fixtures_of_an_infra_test(
    pytester, monkeypatch, target, ran
):
    monkeypatch.setenv("ACCEPTANCE_TARGET", target)
    pytester.makeini("[pytest]\nmarkers =\n    infra: test marker\n")
    pytester.makeconftest(GUARD_CONFTEST)
    pytester.makepyfile(INFRA_TEST)
    result = pytester.runpytest("-p", "no:cacheprovider", "--strict-markers")
    if ran:
        result.assert_outcomes(passed=1)
    else:
        result.assert_outcomes(errors=1)
        result.stdout.fnmatch_lines(
            ["*infra tests run only against the local servers*"]
        )
    assert (pytester.path / "fixture-ran").exists() == ran


PAGE = ["c", "a", "b"]
STATE_BEFORE = {"version": 0, "log_position": 0}
STATE_AFTER = {"version": 2, "log_position": 10000}
TIMES = {
    "created": 100.0,
    "written": 112.5,
    "copy_started": 114.0,
    "copy_finished": 171.25,
}


def test_compaction_properties_of_a_copy_that_crossed_a_renumbering():
    before = {0: PAGE, 5000: ["d", "e"], 10000: ["f"]}
    after = {0: ["a", "c", "b"], 5000: ["d", "e"], 10000: ["g"]}
    properties = e2e.compaction_properties(
        before, after, STATE_BEFORE, STATE_AFTER, TIMES
    )
    assert properties == [
        ("crossed", True),
        ("changed_offsets", [0, 10000]),
        ("state_before", STATE_BEFORE),
        ("state_after", STATE_AFTER),
        ("write_seconds", 12.5),
        ("copy_started_after_create_seconds", 14.0),
        ("copy_finished_after_create_seconds", 71.25),
        ("copy_seconds", 57.25),
    ]


def test_compaction_properties_of_a_copy_whose_page_0_kept_its_order():
    # A later page can change while page 0 does not: then the first
    # compaction run held page 0 and 5,000 records or fewer.
    before = {0: PAGE, 5000: ["d", "e"]}
    after = {0: list(PAGE), 5000: ["e", "d"]}
    properties = dict(
        e2e.compaction_properties(before, after, STATE_BEFORE, STATE_BEFORE, TIMES)
    )
    assert properties["crossed"] is False
    assert properties["changed_offsets"] == [5000]
    assert properties["state_after"] == STATE_BEFORE
