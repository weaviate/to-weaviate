import json
from pathlib import Path

from to_weaviate.sources.chroma.config import distance_of, embedding_of, sparse_keys_of

FIXTURES = Path(__file__).parent / "fixtures"


def load(name):
    return json.loads((FIXTURES / name).read_text())


def test_distance_from_hnsw():
    assert distance_of(load("collection_1x_hnsw.json")) == "cosine"


def test_distance_from_spann():
    assert distance_of(load("collection_1x_spann.json")) == "dot"


def test_distance_legacy_metadata_wins_on_05x():
    assert distance_of(load("collection_05x.json")) == "cosine"


def test_distance_default():
    assert distance_of({"metadata": None, "configuration_json": {}}) == "l2-squared"


def test_distance_missing_configuration():
    assert distance_of({}) == "l2-squared"


def test_distance_l2_maps_to_l2_squared():
    config = {"configuration_json": {"hnsw": {"space": "l2"}}}
    assert distance_of(config) == "l2-squared"


def test_embedding_known():
    assert embedding_of(load("collection_1x_hnsw.json"))["name"] == "openai"


def test_embedding_legacy_is_none():
    config = {"configuration_json": {"embedding_function": {"type": "legacy"}}}
    assert embedding_of(config) is None


def test_embedding_missing_is_none():
    assert embedding_of({"configuration_json": None}) is None


def test_sparse_keys_from_schema():
    schema = {
        "keys": {
            "sp": {"sparse_vector": {"index": {}}},
            "plain": {"string": {}},
            "also": {"sparse_vector": {}},
        }
    }
    assert sparse_keys_of({"schema": schema}) == ["also", "sp"]


def test_sparse_keys_without_schema():
    assert sparse_keys_of({}) == []
    assert sparse_keys_of({"schema": None}) == []
