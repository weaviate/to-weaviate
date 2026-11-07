import math

from chroma.seed import records
from conftest import (
    run,
    weaviate_by_source_id,
    weaviate_config,
    weaviate_near_text,
    weaviate_near_vector,
)

MODULE = "text2vec-transformers"
# The dimension of the model that the acceptance servers serve.
EMBEDDED_DIMENSION = 384
FLOAT32_TOLERANCE = 1e-6


def _inspect_tiny_l2(workdir, vectors: str) -> None:
    r = run(
        [
            "inspect",
            "chroma",
            "--collection",
            "tiny.l2",
            "--vectors",
            vectors,
            "--vectorizer",
            MODULE,
        ],
        workdir,
    )
    assert r.exit_code == 0, r
    for cmd in (["check"], ["copy"], ["verify"]):
        r = run(cmd, workdir)
        assert r.exit_code == 0 and r.data["status"] == "success", r


def test_embed_mode_vectorizes_the_documents_in_weaviate(seeded, tmp_path):
    _inspect_tiny_l2(tmp_path, "embed")
    vectors = weaviate_config("Tiny_l2").vector_config
    assert list(vectors) == ["default"]
    assert vectors["default"].vectorizer.vectorizer.value == MODULE
    for record in records("tiny.l2", limit=3):
        found = weaviate_by_source_id("Tiny_l2", record.id)
        assert found is not None
        # verify passed although the stored vector is not the source vector,
        # so it compared the properties only.
        assert len(found.vector["default"]) == EMBEDDED_DIMENSION
        assert len(record.embedding) != EMBEDDED_DIMENSION
        assert record.document is not None
        nearest = weaviate_near_text("Tiny_l2", record.document)
        assert nearest[0] == record.id, (record.document, nearest)


def test_both_mode_keeps_the_source_vector_and_embeds_the_document(seeded, tmp_path):
    _inspect_tiny_l2(tmp_path, "both")
    vectors = weaviate_config("Tiny_l2").vector_config
    assert sorted(vectors) == ["default", "source"]
    assert vectors["default"].vectorizer.vectorizer.value == MODULE
    assert vectors["source"].vectorizer.vectorizer.value == "none"
    assert vectors["source"].vector_index_config.distance_metric.value == "l2-squared"
    for record in records("tiny.l2", limit=3):
        found = weaviate_by_source_id("Tiny_l2", record.id)
        assert found is not None
        kept = found.vector["source"]
        assert len(kept) == len(record.embedding)
        assert all(
            math.isclose(a, b, abs_tol=FLOAT32_TOLERANCE)
            for a, b in zip(kept, record.embedding, strict=True)
        ), (kept, record.embedding)
        assert len(found.vector["default"]) == EMBEDDED_DIMENSION
        assert record.document is not None
        by_text = weaviate_near_text("Tiny_l2", record.document, "default")
        assert by_text[0] == record.id, (record.document, by_text)
        by_vector = weaviate_near_vector("Tiny_l2", record.embedding, "source")
        assert by_vector[0] == record.id, (record.id, by_vector)
