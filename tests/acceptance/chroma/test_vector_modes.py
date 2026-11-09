import math

from chroma.seed import records
from conftest import (
    copy_and_verify,
    embedded_dimension,
    enabled_vectorizer,
    run,
    weaviate_by_source_id,
    weaviate_config,
    weaviate_near_text,
    weaviate_near_vector,
    weaviate_wait_indexed,
)

FLOAT32_TOLERANCE = 1e-6


def _inspect_tiny_l2(workdir, vectors: str, module: str) -> None:
    r = run(
        [
            "inspect",
            "chroma",
            "--collection",
            "tiny.l2",
            "--vectors",
            vectors,
            "--vectorizer",
            module,
        ],
        workdir,
    )
    assert r.exit_code == 0, r
    copy_and_verify(workdir)


def test_embed_mode_vectorizes_the_documents_in_weaviate(seeded, tmp_path):
    module = enabled_vectorizer()
    _inspect_tiny_l2(tmp_path, "embed", module)
    vectors = weaviate_config("Tiny_l2").vector_config
    assert list(vectors) == ["default"]
    assert vectors["default"].vectorizer.vectorizer.value == module
    weaviate_wait_indexed("Tiny_l2")
    dimension = embedded_dimension("Tiny_l2")
    for record in records("tiny.l2", limit=3):
        found = weaviate_by_source_id("Tiny_l2", record.id)
        assert found is not None
        # verify passed although the stored vector is not the source vector,
        # so it compared the properties only.
        assert len(found.vector["default"]) == dimension
        assert len(record.embedding) != dimension
        assert record.document is not None
        nearest = weaviate_near_text("Tiny_l2", record.document)
        assert nearest[0] == record.id, (record.document, nearest)


def test_both_mode_keeps_the_source_vector_and_embeds_the_document(seeded, tmp_path):
    module = enabled_vectorizer()
    _inspect_tiny_l2(tmp_path, "both", module)
    vectors = weaviate_config("Tiny_l2").vector_config
    assert sorted(vectors) == ["default", "source"]
    assert vectors["default"].vectorizer.vectorizer.value == module
    assert vectors["source"].vectorizer.vectorizer.value == "none"
    assert vectors["source"].vector_index_config.distance_metric.value == "l2-squared"
    weaviate_wait_indexed("Tiny_l2")
    dimension = embedded_dimension("Tiny_l2")
    for record in records("tiny.l2", limit=3):
        found = weaviate_by_source_id("Tiny_l2", record.id)
        assert found is not None
        kept = found.vector["source"]
        assert len(kept) == len(record.embedding)
        assert all(
            math.isclose(a, b, abs_tol=FLOAT32_TOLERANCE)
            for a, b in zip(kept, record.embedding, strict=True)
        ), (kept, record.embedding)
        assert len(found.vector["default"]) == dimension
        assert len(record.embedding) != dimension
        assert record.document is not None
        by_text = weaviate_near_text("Tiny_l2", record.document, "default")
        assert by_text[0] == record.id, (record.document, by_text)
        by_vector = weaviate_near_vector("Tiny_l2", record.embedding, "source")
        assert by_vector[0] == record.id, (record.id, by_vector)


def test_both_mode_keeps_the_source_vector_of_a_record_without_a_document(
    seeded, tmp_path
):
    module = enabled_vectorizer()
    args = ["--collection", "docs-cosine", "--vectors", "both", "--vectorizer", module]
    r = run(["inspect", "chroma", *args], tmp_path)
    assert r.exit_code == 0, r
    copy_and_verify(tmp_path)
    r = run(["status"], tmp_path)
    assert r.data["collections"][0]["rejects"] == 0, r
    without = [record for record in records("docs-cosine", 700) if not record.document]
    assert len(without) == 20
    for record in without:
        found = weaviate_by_source_id("Docs_cosine", record.id)
        assert found is not None and sorted(found.vector) == ["source"], record.id
    first = without[0]
    weaviate_wait_indexed("Docs_cosine")
    by_vector = weaviate_near_vector("Docs_cosine", first.embedding, "source")
    assert by_vector[0] == first.id, (first.id, by_vector)
