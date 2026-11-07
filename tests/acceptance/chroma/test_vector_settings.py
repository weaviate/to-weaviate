import shutil
import time
from pathlib import Path
from typing import Any

import pytest
from chroma.seed import records
from conftest import (
    read_plan,
    run,
    weaviate_compressed,
    weaviate_config,
    weaviate_exists,
    weaviate_near_vector,
    write_plan,
)
from weaviate.collections.classes.config import (
    _BQConfig,
    _PQConfig,
    _RQConfig,
    _SQConfig,
)

# (source collection, target collection, Weaviate distance)
DISTANCES = [
    ("tiny.l2", "Tiny_l2", "l2-squared"),
    ("docs-cosine", "Docs_cosine", "cosine"),
    ("docs-ip", "Docs_ip", "dot"),
]
QUANTIZER_CLASSES = {"rq": _RQConfig, "sq": _SQConfig, "pq": _PQConfig, "bq": _BQConfig}
# docs-rq has 300 records of 32 dimensions, so a training limit of 100 is
# reached and 8 segments divide the dimension.
QUANTIZERS = {
    "default": {"type": "default"},
    "none": {"type": "none"},
    "rq": {"type": "rq", "bits": 8, "rescore_limit": 40, "training_limit": 100},
    "sq": {"type": "sq", "rescore_limit": 25, "training_limit": 100},
    "pq": {"type": "pq", "segments": 8, "centroids": 16, "training_limit": 100},
    "bq": {"type": "bq"},
}
# Quantizers that compress the vectors when the collection is created.
COMPRESS_AT_ONCE = {"rq": True, "bq": True, "none": False}
DYNAMIC_REASON = (
    "B8 finding, not fixed: Weaviate 1.40.4 refuses a dynamic index unless "
    "ASYNC_INDEXING is on, which the acceptance servers do not set, and check "
    "has no way to read that setting, so check passes and copy fails"
)
INDEXES = {
    "default": {"type": "default"},
    "hnsw": {
        "type": "hnsw",
        "options": {"ef": 64, "ef_construction": 96, "max_connections": 24},
        "quantizer": {"type": "sq", "rescore_limit": 25, "training_limit": 100},
    },
    "flat": {
        "type": "flat",
        "options": {"vector_cache_max_objects": 5000},
        "quantizer": {"type": "bq", "rescore_limit": 50, "cache": True},
    },
    "hfresh": {
        "type": "hfresh",
        "options": {"max_posting_size_kb": 64, "replicas": 2, "search_probe": 32},
        "quantizer": {"type": "rq", "bits": 1, "rescore_limit": 66},
    },
    "dynamic": pytest.param(
        {
            "type": "dynamic",
            "options": {"threshold": 1000},
            "quantizer": {"type": "rq", "bits": 8, "rescore_limit": 40},
        },
        marks=pytest.mark.xfail(strict=True, reason=DYNAMIC_REASON),
    ),
}
# Settings that the plan model accepts and that Weaviate 1.40.4 refuses, or
# creates with other values than the plan holds.
UNSUPPORTED = {
    "flat-sq": {"type": "flat", "quantizer": {"type": "sq"}},
    "flat-pq": {"type": "flat", "quantizer": {"type": "pq"}},
    "flat-rq-4-bits": {"type": "flat", "quantizer": {"type": "rq", "bits": 4}},
    "flat-rq-centering": {
        "type": "flat",
        "quantizer": {"type": "rq", "bits": 8, "centering": True},
    },
    "flat-rq-training-limit": {
        "type": "flat",
        "quantizer": {"type": "rq", "training_limit": 100},
    },
    "hfresh-none": {"type": "hfresh", "quantizer": {"type": "none"}},
    "hfresh-sq": {"type": "hfresh", "quantizer": {"type": "sq"}},
    "hfresh-pq": {"type": "hfresh", "quantizer": {"type": "pq"}},
    "hfresh-bq": {"type": "hfresh", "quantizer": {"type": "bq"}},
    "hfresh-rq-8-bits": {"type": "hfresh", "quantizer": {"type": "rq", "bits": 8}},
    "hfresh-rq-centering": {
        "type": "hfresh",
        "quantizer": {"type": "rq", "bits": 1, "centering": True},
    },
    "hfresh-rq-cache": {"type": "hfresh", "quantizer": {"type": "rq", "cache": True}},
    "hfresh-rq-training-limit": {
        "type": "hfresh",
        "quantizer": {"type": "rq", "training_limit": 100},
    },
    "hnsw-rq-centering-8-bits": {
        "type": "hnsw",
        "quantizer": {"type": "rq", "bits": 8, "centering": True},
    },
    "hnsw-rq-cache": {"type": "hnsw", "quantizer": {"type": "rq", "cache": True}},
    "hnsw-bq-cache": {"type": "hnsw", "quantizer": {"type": "bq", "cache": True}},
    "hnsw-bq-rescore-limit": {
        "type": "hnsw",
        "quantizer": {"type": "bq", "rescore_limit": 30},
    },
    "hnsw-pq-segments-not-a-divisor": {
        "type": "hnsw",
        "quantizer": {"type": "pq", "segments": 5, "training_limit": 100},
    },
}
COMPRESSION_REASON = (
    "B8 finding, not fixed: Weaviate 1.40.4 without ASYNC_INDEXING never "
    "trains a quantizer that the collection had when it was created, and copy "
    "creates the collection with it"
)
TRAINED = {
    "sq": {"type": "sq", "training_limit": 100},
    "pq": {"type": "pq", "segments": 8, "centroids": 16, "training_limit": 100},
    "rq-centering": {
        "type": "rq",
        "bits": 4,
        "centering": True,
        "training_limit": 100,
    },
}
COMPRESSION_WAIT = 3.0


def _copy_and_verify(workdir: Path) -> None:
    for cmd in (["check"], ["copy"], ["verify"]):
        r = run(cmd, workdir)
        assert r.exit_code == 0 and r.data["status"] == "success", r


def test_chroma_distances_map_to_weaviate_distances(seeded, tmp_path):
    names = [arg for source, _, _ in DISTANCES for arg in ("--collection", source)]
    r = run(["inspect", "chroma", *names], tmp_path)
    assert r.exit_code == 0, r
    _copy_and_verify(tmp_path)
    for source, target, distance in DISTANCES:
        index = weaviate_config(target).vector_config["default"].vector_index_config
        assert index.distance_metric.value == distance, target
        for record in records(source, limit=3):
            nearest = weaviate_near_vector(target, record.embedding)
            assert nearest[0] == record.id, (target, record.id, nearest)


@pytest.fixture(scope="module")
def docs_rq_inspection(seeded, tmp_path_factory) -> Path:
    """A directory with the inspection of docs-rq and its drafted plan."""
    workdir = tmp_path_factory.mktemp("docs-rq")
    r = run(["inspect", "chroma", "--collection", "docs-rq"], workdir)
    assert r.exit_code == 0, r
    return workdir


def _plan_with_index(inspection: Path, workdir: Path, index: dict[str, Any]) -> None:
    """Write the docs-rq inspection to `workdir`, with `index` in the plan."""
    shutil.copy(inspection / "inspect.json", workdir / "inspect.json")
    plan = read_plan(inspection)
    plan["collections"][0]["target"]["vectors"][0]["index"] = index
    write_plan(workdir, plan)


def _assert_quantizer(found: Any, planned: dict[str, Any]) -> None:
    kind = planned["type"]
    if kind == "none":
        assert found is None
    elif kind != "default":
        assert isinstance(found, QUANTIZER_CLASSES[kind]), found
        for field, value in planned.items():
            if field != "type":
                assert getattr(found, field) == value, (field, found)


def _docs_rq_index() -> Any:
    return weaviate_config("Docs_rq").vector_config["default"].vector_index_config


@pytest.mark.parametrize("quantizer", QUANTIZERS.values(), ids=QUANTIZERS)
def test_quantizer_settings_reach_the_server(docs_rq_inspection, tmp_path, quantizer):
    index = {"type": "hnsw", "quantizer": quantizer}
    _plan_with_index(docs_rq_inspection, tmp_path, index)
    _copy_and_verify(tmp_path)
    found = _docs_rq_index()
    assert found.vector_index_type() == "hnsw"
    _assert_quantizer(found.quantizer, quantizer)
    if quantizer["type"] in COMPRESS_AT_ONCE:
        expected = COMPRESS_AT_ONCE[quantizer["type"]]
        assert weaviate_compressed("Docs_rq") == [expected]


@pytest.mark.parametrize("index", INDEXES.values(), ids=INDEXES)
def test_index_settings_reach_the_server(docs_rq_inspection, tmp_path, index):
    _plan_with_index(docs_rq_inspection, tmp_path, index)
    _copy_and_verify(tmp_path)
    found = _docs_rq_index()
    assert found.distance_metric.value == "cosine"
    if index["type"] == "default":
        return
    assert found.vector_index_type() == index["type"]
    for option, value in index["options"].items():
        assert getattr(found, option) == value, (option, found)
    parts = [found.hnsw, found.flat] if index["type"] == "dynamic" else [found]
    for part in parts:
        _assert_quantizer(part.quantizer, index["quantizer"])


@pytest.mark.parametrize("index", UNSUPPORTED.values(), ids=UNSUPPORTED)
def test_check_refuses_index_settings_that_the_server_does_not_create(
    docs_rq_inspection, tmp_path, index
):
    _plan_with_index(docs_rq_inspection, tmp_path, index)
    for cmd in (["check"], ["copy"]):
        r = run(cmd, tmp_path)
        assert r.exit_code == 2 and r.data["code"] == "invalid_plan", r
    assert not weaviate_exists("Docs_rq")


@pytest.mark.xfail(strict=True, reason=COMPRESSION_REASON)
@pytest.mark.parametrize("quantizer", TRAINED.values(), ids=TRAINED)
def test_trained_quantizer_compresses_the_vectors(
    docs_rq_inspection, tmp_path, quantizer
):
    index = {"type": "hnsw", "quantizer": quantizer}
    _plan_with_index(docs_rq_inspection, tmp_path, index)
    _copy_and_verify(tmp_path)
    deadline = time.monotonic() + COMPRESSION_WAIT
    while weaviate_compressed("Docs_rq") != [True] and time.monotonic() < deadline:
        time.sleep(0.5)
    assert weaviate_compressed("Docs_rq") == [True]
