# Testing

The tests check that a migration loses no record, writes none twice, leaks
no secret, resumes after a killed process or a refused write, and reports
each failure with the right exit code. They run at three levels.

| Level | Where | Servers | Runs in CI |
|---|---|---|---|
| Unit | `tests/unit/` | None. Fakes stand in for the source and Weaviate. | Yes, on Python 3.10, 3.13 and 3.14 |
| Acceptance, API | `tests/acceptance/` | A real source and a real Weaviate, reached only through the `to-weaviate` command and the server APIs | On pull requests and releases |
| Acceptance, infra | `tests/acceptance/`, marker `infra` | Also restarts or deletes a server through Docker or kubectl | On pull requests and releases |

The API level also runs against a Weaviate Cloud cluster with
`make acceptance-cloud`, by hand before a release. See
[Against Weaviate Cloud](#against-weaviate-cloud).

Acceptance tests exist for Chroma only, in `tests/acceptance/chroma/`. Each
new source adds its own directory, `tests/acceptance/<name>/`. The shared
pieces are `tests/acceptance/conftest.py` and `tests/e2e.py`.

## What the tests cover

| Area | What the tests check |
|---|---|
| Lost or duplicated records | Faults in the middle of a page, short pages, refused batches, objects above the gRPC message limit, two source ids with one UUID, a second `copy` on the same directory, a 50,000-record copy |
| Resuming | A killed `copy`, a refused page, a refused quantizer update, a state file of an older or newer format, a corrupted state file, a changed target or source |
| Sync | Records added, changed and deleted after the copy, `--prune` and its 10% limit, a source written to during the id listing, one tenant of a shared collection with an object the tool did not write, a plan edit after the copy, rejects written once the plan is fixed |
| Secrets | No API key, URL password or embedding-function secret in any output, traceback or file of the migration directory. The acceptance suite checks stdout and stderr of each command that it runs to completion, and every file under the test's directory. A command started in the background (`start` in `tests/e2e.py`) has its stderr discarded. |
| Plan and schema | Every plan field, the allowed edits end to end, the index and quantizer rules, property types and conversions, names that need renaming, existing collections with other settings |
| Data fidelity | Distances, metadata types, values at the type limits, unicode, empty and long values, embedding-function mappings |
| Exit codes and JSON | Every error code and its exit code, one JSON document on stdout, text mode |
| Faults on a 3-node cluster | A Weaviate pod deleted during `copy` and during `verify`, each consistency level on 3 replicas |
| Source behavior | For Chroma: the page order, a Chroma restart during a copy, and a copy while distributed Chroma compacts |

## Not covered

- Chroma Cloud. No test account exists. The unit tests cover the tool's side
  with fakes, and the `cluster` setup of CI covers distributed
  Chroma, which Chroma Cloud runs. Its quotas, rate limits and sharded
  collections stay untested.
- A Weaviate Cloud sandbox cluster, and Weaviate versions other than
  `tests/acceptance/weaviate-version`.
- A Weaviate key that may read but not write. Unit tests cover the error.
- Faults that need a proxy: a connection reset inside a response body, or a
  server slow enough to hit a timeout. Unit tests inject them.
- Values that Chroma refuses to store, such as vectors of mixed dimension or
  ints outside int64. Unit tests cover the tool's side.
- Vectorizers that call a model provider. CI has no provider keys.
- Windows, Ctrl-C, and sources of millions of records.

## Unit tests

```bash
.venv/bin/python -m pytest tests/unit
```

`tests/unit/fakes.py` holds `FakeSource` and the fake Weaviate writer, reader
and schema. `tests/unit/chroma_server.py` serves a fake Chroma through
`httpx.MockTransport`, so the CLI tests run the real Chroma client.

## Acceptance tests

The acceptance tests run the installed `to-weaviate` command against real
servers that are already running. No test starts a server. Some `infra`
tests restart the compose Chroma or delete a Weaviate pod. The tests run
only against the test endpoints:

```bash
WEAVIATE_URL=http://localhost:8099
WEAVIATE_GRPC_URL=http://localhost:50099
WEAVIATE_API_KEY=admin-key
CHROMA_URL=http://localhost:8001
```

When a variable is missing or has another value, the suite skips every test
and names the variable. With `TO_WEAVIATE_REQUIRE_ACCEPTANCE=1`, as in CI,
it fails instead. It connects to no server before this check passes. The
commands it runs get empty `CHROMA_DATABASE`, `CHROMA_TENANT`,
`CHROMA_API_KEY` and `CHROMA_TOKEN_HEADER`, so values from your shell do not
reach them.

The suite deletes and recreates the Chroma collections that `seed.py`
creates. Before and after each test it deletes the Weaviate collections in
`SUITE_COLLECTIONS` (`tests/acceptance/conftest.py`), and it writes to no
other collection.

`pytest -m "not infra"` runs the API level only.

### With Docker

```bash
make chroma-up       # Chroma on port 8001
make weaviate-up     # Weaviate and text2vec-transformers on ports 8099 and 50099
source .venv/bin/activate
WEAVIATE_ASYNC_INDEXING=true make acceptance
make weaviate-down
make chroma-down
```

`make weaviate-up` starts Weaviate with `ASYNC_INDEXING=true`, as Weaviate
Cloud runs. `ASYNC=0 make weaviate-up` starts it without; then run the suite
with `WEAVIATE_ASYNC_INDEXING=false make acceptance`. A few tests depend on
async indexing and read this variable.

### With a kind cluster

`make acceptance-up` starts Chroma and a Weaviate kind cluster named
`weaviate-k8s` through a clone of
[weaviate-local-k8s](https://github.com/weaviate/weaviate-local-k8s) at
`LOCAL_K8S` (default: `../weaviate-local-k8s`). It needs Docker, `kind`,
`helm`, `kubectl`, `curl`, `jq`, `lsof`, `nohup`, and `gtimeout` on macOS or
`timeout` on Linux. It stops when a kind cluster with that name exists, and
it switches the kubectl context to `kind-weaviate-k8s`.

```bash
make acceptance-up
source .venv/bin/activate
make acceptance
make acceptance-down   # stops Chroma, then asks before it deletes the cluster
```

`make acceptance-down` deletes only the cluster that `make acceptance-up`
created, and only after you type `y`. The cleanup of weaviate-local-k8s also
kills every `kubectl-relay` process on the machine and leaves no kubectl
context selected.

### Against Weaviate Cloud

`make acceptance-cloud` runs the API level against a Weaviate Cloud cluster,
with the local test Chroma as the source. Use a dedicated serverless
cluster: the suite fails before its first test when the cluster holds a
collection outside `SUITE_COLLECTIONS`. The cluster must have
`text2vec-weaviate` enabled and at least one of the `CANDIDATE_MODULES` of
`tests/acceptance/chroma/test_check.py` disabled.

Put the two `export` lines in a file outside the repository, so the key stays
out of your shell history, and `source` it:

```bash
make chroma-up
source ~/weaviate-cloud-test.sh   # export WEAVIATE_URL=... and export WEAVIATE_API_KEY=...
source .venv/bin/activate
make acceptance-cloud
```

Do not set the two variables on the `make` command line. Make passes such
values to every child process, and the target refuses them.

## CI

`.github/workflows/ci.yml` runs these jobs:

| Job | What it runs | Pull request | Push to `main` | Manual run |
|---|---|---|---|---|
| `lint` | `pre-commit run --all-files`: ruff, ruff-format, mypy | Yes | Yes | Yes |
| `unit` | `pytest tests/unit` on Python 3.10, 3.13 and 3.14 | Yes | Yes | Yes |
| `unit (lowest deps)` | `pytest tests/unit` on Python 3.10 with the lower bound of each dependency in `pyproject.toml` | Yes | Yes | Yes |
| `acceptance` | Both acceptance setups, through `.github/workflows/acceptance-suites.yml` | Yes | No | Yes |

A manual run is "Run workflow" on the CI workflow page, or
`gh workflow run ci.yml --ref <branch>`. `.github/workflows/release.yml`
runs the `acceptance` job on the tagged commit. See [Releasing](releasing.md).

Every job except `unit (lowest deps)` installs the versions in `uv.lock` with
`uv sync --locked --extra test`, which fails when `uv.lock` does not match
`pyproject.toml`.

The `acceptance` job has two setups. Each runs Weaviate in a kind cluster
with `ASYNC_INDEXING`.

| Setup | Servers | Check that holds its result |
|---|---|---|
| `single` | One Weaviate node and the single-node Chroma | `acceptance / single / run` |
| `cluster` | Three Weaviate nodes with replication factor 3 and distributed Chroma | `acceptance / cluster result` |

The tests in `tests/acceptance/chroma/test_cluster.py` run in the `cluster`
setup only. Each setup uploads its JUnit XML as the artifact
`acceptance-junit-single` or `acceptance-junit-cluster`.
`test_copy_during_compaction` records its measurements there as test
properties.

### The cluster result

Distributed Chroma needs a CPU with AVX-512, and part of GitHub's
`ubuntu-latest` runners lack it. No runner label guarantees it. The
`cluster` setup therefore makes up to four attempts, one after the other,
each on a runner that GitHub picks: `acceptance / cluster runner attempt 1 / run`
to `attempt 4`. An attempt on a runner without AVX-512 ends in seconds, runs
no test and is green, and the next attempt starts. The first attempt on a
runner with AVX-512 runs the suite, and the later attempts are skipped.

A green or skipped attempt therefore says nothing about the suite. Read
`acceptance / cluster result`. It passes only when exactly one attempt ran
the suite and that run passed. It fails when the suite failed, and it fails
when no attempt ran it.

When it reports that no attempt got a runner with AVX-512, attempt 4 has
failed too, and no test ran. Rerun the failed jobs, which runs attempt 4 on
another runner and then the result job again:

```bash
gh run rerun <run id> --failed
```

Repeat it until an attempt gets such a runner.

## Writing tests

- A behavior change comes with a unit test that fails without it.
- An acceptance test reaches the servers only through the command and the
  server APIs, unless it needs to restart or delete a server. Then it
  carries the `infra` marker.
- Test data for a source lives in its `seed.py`. A new target collection
  goes into `SUITE_COLLECTIONS`, so the suite cleans it up.
- Use made-up keys in tests, never real ones.
