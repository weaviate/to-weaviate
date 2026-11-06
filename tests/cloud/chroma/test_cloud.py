"""The cloud suite: Chroma Cloud to Weaviate Cloud.

The test runs the installed to-weaviate command against the servers that the
environment names. It starts no server and passes the environment to the
command unchanged. It connects to no server before the variable check passes.
"""

import os
from collections.abc import Iterator

import pytest
from e2e import close_weaviate, delete_target_collection, run, weaviate_count

REQUIRE_VARIABLE = "TO_WEAVIATE_REQUIRE_CLOUD"
REQUIRED_VARIABLES = ("WEAVIATE_URL", "WEAVIATE_API_KEY")
# One of the two is enough: a Chroma server without authentication needs no key.
CHROMA_VARIABLES = ("CHROMA_API_KEY", "CHROMA_URL")
# The target collection of the Chroma collection to-weaviate-ci. Only this
# suite uses the name, so the test deletes it before and after the run.
TARGET = "To_weaviate_ci"


def _is_set(name: str) -> bool:
    return bool(os.environ.get(name, "").strip())


def missing_variables() -> list[str]:
    """The required variables that are not set. No value appears in the text."""
    missing = [name for name in REQUIRED_VARIABLES if not _is_set(name)]
    if not any(_is_set(name) for name in CHROMA_VARIABLES):
        missing.append(" or ".join(CHROMA_VARIABLES))
    return missing


# This fixture is not in a conftest.py: test_migration.py imports from the
# module `conftest`, and a second conftest.py outside a package can take that
# module name.
@pytest.fixture(scope="module", autouse=True)
def _cloud_variables() -> Iterator[None]:
    missing = missing_variables()
    if missing:
        reason = "the cloud suite needs these variables: " + "; ".join(missing)
        # The release job sets it, so that publish never runs after a skip.
        if os.environ.get(REQUIRE_VARIABLE) == "1":
            pytest.fail(reason)
        pytest.skip(reason)
    yield
    close_weaviate()


def test_chroma_cloud_to_weaviate_cloud(tmp_path):
    delete_target_collection(TARGET)  # test code; the tool never deletes a collection
    try:
        r = run(["inspect", "chroma", "--collection", "to-weaviate-ci"], tmp_path)
        assert r.exit_code == 0, r
        # copy must write to the collection that the test deletes.
        assert [c["target"] for c in r.data["collections"]] == [TARGET], r
        for cmd in (["check"], ["copy"], ["verify"]):
            r = run(cmd, tmp_path)
            assert r.exit_code == 0 and r.data["status"] == "success", r
        assert weaviate_count(TARGET) == 2000
    finally:
        delete_target_collection(TARGET)
