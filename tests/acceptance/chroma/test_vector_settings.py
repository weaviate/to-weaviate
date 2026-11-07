from chroma.seed import records
from conftest import run, weaviate_config, weaviate_near_vector

# (source collection, target collection, Weaviate distance)
DISTANCES = [
    ("tiny.l2", "Tiny_l2", "l2-squared"),
    ("docs-cosine", "Docs_cosine", "cosine"),
    ("docs-ip", "Docs_ip", "dot"),
]


def _copy_and_verify(args: list[str], workdir) -> None:
    r = run(["inspect", "chroma", *args], workdir)
    assert r.exit_code == 0, r
    for cmd in (["check"], ["copy"], ["verify"]):
        r = run(cmd, workdir)
        assert r.exit_code == 0 and r.data["status"] == "success", r


def test_chroma_distances_map_to_weaviate_distances(seeded, tmp_path):
    _copy_and_verify(
        [arg for source, _, _ in DISTANCES for arg in ("--collection", source)],
        tmp_path,
    )
    for source, target, distance in DISTANCES:
        index = weaviate_config(target).vector_config["default"].vector_index_config
        assert index.distance_metric.value == distance, target
        for record in records(source, limit=3):
            nearest = weaviate_near_vector(target, record.embedding)
            assert nearest[0] == record.id, (target, record.id, nearest)
