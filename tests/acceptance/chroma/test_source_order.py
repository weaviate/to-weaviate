"""The order of `get` with offset and limit, which copy relies on.

copy reads a collection in offset partitions, resumes at an offset, and finds
a shift of the source by the ids at the end of each partition. That needs
`get` to return the records in insertion order: a deletion moves each later
record one offset earlier, and an addition goes last. On a distributed Chroma
the first case waits until each write is compacted, as the seed does. The
second case copies while distributed Chroma compacts, without that wait.
"""

import time

import pytest
from chroma.seed import (
    ADD_BATCH,
    BURST,
    BURST_COUNT,
    ORDER,
    ORDER_COUNT,
    add_records,
    burst_records,
    collection_state,
    count,
    delete_collection,
    delete_records,
    make_collection,
    order_records,
    page_ids,
    wait_indexed,
)
from conftest import chroma_kind, compaction_properties, run

from to_weaviate.sources.chroma.client import CHROMA_MAX_PAGE

# The positions in ORDER of the records that the case deletes.
DELETED = (400, 450, 500, 550, 600)
ADDED = 10
# max_partition_size of the worker config of distributed Chroma: a compaction
# run of more records gives them new offsets. A run takes records of
# contiguous offsets, so a longer run holds the start of a page at a multiple
# of it.
PARTITION_SIZE = 5000


def _written(name: str) -> None:
    """On a distributed Chroma, wait until the writes to `name` are compacted."""
    if chroma_kind() == "distributed":
        wait_indexed([name])


def _ids(name: str) -> list[str]:
    """The ids of `name`, read with `get` from offset 0 in pages of CHROMA_MAX_PAGE."""
    found: list[str] = []
    for offset in range(0, count(name), CHROMA_MAX_PAGE):
        found.extend(page_ids(name, offset, CHROMA_MAX_PAGE))
    return found


def _pages(name: str) -> dict[int, list[str]]:
    """The ids of the pages of `name` that start at the multiples of PARTITION_SIZE."""
    return {
        offset: page_ids(name, offset, CHROMA_MAX_PAGE)
        for offset in range(0, BURST_COUNT, PARTITION_SIZE)
    }


def test_get_pages_follow_the_insertion_order():
    records = order_records(0, ORDER_COUNT)
    expected = [r.id for r in records]
    # Otherwise an order by id would pass too.
    assert expected != sorted(expected)
    kind = chroma_kind()
    try:
        make_collection(ORDER, records, ADD_BATCH)
        _written(ORDER)
        assert _ids(ORDER) == expected, f"{ORDER} on {kind} Chroma after its adds"

        deleted = [expected[i] for i in DELETED]
        delete_records(ORDER, deleted)
        _written(ORDER)
        expected = [i for i in expected if i not in deleted]
        assert (
            _ids(ORDER) == expected
        ), f"{ORDER} on {kind} Chroma after the deletion of {deleted}"

        added = order_records(ORDER_COUNT, ADDED)
        add_records(ORDER, added)
        _written(ORDER)
        expected += [r.id for r in added]
        assert (
            _ids(ORDER) == expected
        ), f"{ORDER} on {kind} Chroma after the addition of {ADDED} records"
    finally:
        delete_collection(ORDER)


@pytest.mark.skipif(
    chroma_kind() != "distributed",
    reason="needs ACCEPTANCE_CHROMA=distributed (the cluster setup of CI), whose "
    f"compaction numbers records again; it is {chroma_kind()}",
)
def test_copy_during_compaction(tmp_path, request):
    """copy and verify a collection right after a burst of writes.

    Distributed Chroma compacts every 10 s, at most 10,000 records of a
    collection per run. A run of more than 5,000 records gives them new
    offsets in another order. The frontend serves `get` from a collection
    cache whose entries live 60 s, so a renumbering becomes visible at a
    cache expiry, not when the run ends. The case records, as test
    properties in the JUnit XML, whether the copy ran across such an expiry:
    `crossed` is whether page 0 of burst changed between right before and
    right after the copy. `changed_offsets` lists the offsets of the pages,
    one every 5,000 offsets, that changed. A pass with `crossed` false and
    no changed offset shows nothing about compaction. An offset other than
    0 shows that a later run's renumbering became visible during the copy.
    A pass with `crossed` true, or such an offset, and a clean verify is
    evidence that copy survives the renumbering. A failure with `crossed`
    true, or such an offset, is the bug that the compaction guard would fix.
    """
    times = {"created": time.monotonic()}
    try:
        make_collection(BURST, burst_records(), ADD_BATCH)
        times["written"] = time.monotonic()
        r = run(
            ["inspect", "chroma", "--collection", BURST, "--sample-pages", "1"],
            tmp_path,
        )
        assert r.exit_code == 0, r
        pages_before, state_before = _pages(BURST), collection_state(BURST)
        times["copy_started"] = time.monotonic()
        r = run(["copy", "--parallelism", "8"], tmp_path)
        times["copy_finished"] = time.monotonic()
        pages_after, state_after = _pages(BURST), collection_state(BURST)
        properties = compaction_properties(
            pages_before, pages_after, state_before, state_after, times
        )
        request.node.user_properties.extend(properties)
        recorded = dict(properties)
        assert r.exit_code == 0 and r.data["status"] == "success", (r, recorded)
        r = run(["verify"], tmp_path)
        assert "checks" in r.data, (r, recorded)
        checks = {c["name"]: c["detail"] for c in r.data["checks"]}
        expected = f"source count {BURST_COUNT}, target count {BURST_COUNT}"
        assert checks["count"] == expected, (checks, recorded)
        assert checks["ids"] == "missing 0, extra 0", (checks, recorded)
        assert r.exit_code == 0, (r, recorded)
    finally:
        delete_collection(BURST)
