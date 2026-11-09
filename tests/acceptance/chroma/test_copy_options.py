import pytest
from conftest import run, weaviate_count


@pytest.mark.parametrize("level", ["ALL", "QUORUM"])
def test_copy_with_a_consistency_level(seeded, tmp_path, level):
    r = run(["inspect", "chroma", "--collection", "tiny.l2"], tmp_path)
    assert r.exit_code == 0, r
    r = run(["copy", "--consistency", level], tmp_path)
    assert r.exit_code == 0, r
    assert weaviate_count("Tiny_l2") == 10


# (copy --parallelism, verify --sample, records the sample holds)
BOUNDS = [(1, 1, 1), (8, 10000, 700)]


@pytest.mark.parametrize(("parallelism", "sample", "sampled"), BOUNDS)
def test_copy_and_verify_at_the_bounds_of_their_options(
    seeded, tmp_path, parallelism, sample, sampled
):
    r = run(["inspect", "chroma", "--collection", "docs-cosine"], tmp_path)
    assert r.exit_code == 0, r
    r = run(["copy", "--parallelism", str(parallelism)], tmp_path)
    assert r.exit_code == 0, r
    assert weaviate_count("Docs_cosine") == 700
    r = run(["verify", "--sample", str(sample)], tmp_path)
    assert r.exit_code == 0, r
    records = next(c for c in r.data["checks"] if c["name"] == "records")
    assert records["detail"].startswith(f"sampled {sampled}, compared {sampled},")
