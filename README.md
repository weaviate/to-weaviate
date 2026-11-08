# to-weaviate

`to-weaviate` copies collections from Chroma into Weaviate. It reads each
record's id, vector, document, URI and metadata from Chroma and writes the
records as Weaviate objects. The tool never writes to or deletes from the
source.

## Chroma Cloud dates

Chroma Cloud stops accepting writes on 2026-12-07 and stops serving reads on
2026-12-22. Run the migration from Chroma Cloud before 2026-12-22. After that
date you can migrate only from a Chroma server that you run yourself, for
example one that serves a local Chroma directory.

## Install

From a clone of this repository:

```bash
pip install .
```

## Quickstart

Put the credentials in a `.env` file in the directory where you run the
tool, or export them:

```
CHROMA_API_KEY=<Chroma Cloud API key>
WEAVIATE_URL=<Weaviate Cloud cluster URL>
WEAVIATE_API_KEY=<Weaviate API key>
```

Then run the commands in this order:

```bash
to-weaviate inspect chroma   # writes .to-weaviate/inspect.json and .to-weaviate/plan.yaml
                             # review and edit .to-weaviate/plan.yaml
to-weaviate check            # checks the plan against Weaviate
to-weaviate copy             # creates the collections and copies the records
to-weaviate verify           # compares Weaviate with Chroma
to-weaviate status           # shows the copy progress at any time
```

## Commands

### `inspect chroma`

Reads the Chroma collections and writes `inspect.json` and a draft
`plan.yaml`. The plan names each target collection, its properties and its
vector settings. Review it before you run `check`.

- `--database NAME`, `--collection NAME`: limit the inspection. Both repeat.
- `--sample-pages N`: metadata pages to read per collection. Default: all.
- `--vectors keep|embed|both`: keep the Chroma vectors (default), let a
  Weaviate vectorizer embed the documents, or both. With `both`, a record
  without a document keeps its source vector and gets no embedded vector.
- `--vectorizer MODULE`, `--vectorizer-option KEY=VALUE`: the Weaviate
  vectorizer module and its options. `plan.yaml` holds each value as text;
  `copy` sends it as a number or a boolean where the module takes one.
- `--index`, `--quantizer`, `--quantizer-bits`: the vector index type, the
  quantizer and the bits for `rq` (1, 4 or 8). `default` leaves the choice
  to the server. `hnsw` takes every quantizer, `flat` and `dynamic` take
  `none`, `rq` and `bq`, and `hfresh` takes `rq` only, as Weaviate 1.40.4
  does. `rq` takes 1, 4 or 8 bits with `hnsw`, 1 or 8 with `flat` and
  `dynamic`, and 1 with `hfresh`; its `centering` needs `hnsw` and 4 bits.
  A quantizer field in `plan.yaml` that the index type does not apply is
  refused too: `cache` of `rq` and `bq`, and `rescore_limit` of `bq`, with
  `hnsw`; `training_limit` of `rq` with `flat`; `cache` and
  `training_limit` of `rq` with `hfresh`. `inspect`, `check` and `copy`
  refuse any other setting. `check` and `copy` also refuse `pq` `segments`
  that do not divide the dimension of the source vectors.
- `--force`: replace an existing `plan.yaml`.

### `check`

Checks `plan.yaml` against the Weaviate server: the plan itself, the
vectorizer modules, and existing collections with other settings. It
creates nothing. A collection, property or tenant name that Weaviate
refuses is the error `invalid_name`; a tenant name has 1 to 64 letters,
digits, `_` or `-`. A planned property that an existing collection lacks is
the warning `missing_property`: `copy` adds it to the collection. A
trained quantizer whose `training_limit` is above the record count of the
source is the warning `quantizer_training`. Plan
entries that write into one collection, each into its own tenant, must
have the same vectors and the same settings for each property that more
than one of them has, and no two property names that differ only in case.

### `copy`

Creates the target collections, then copies the records. Progress goes to
`state.sqlite`, so a second run continues where the first one stopped.
Each run also copies the records added to the source since the last run:
it reads the records after the last two that the run before read. When
deletions moved those two records, the run finds them at their new offsets
and reads what follows. When one of them is gone, the run reads again from
an earlier record or from the start, with the warning "the source changed
since the last run; re-reading from the start"; a record read again is
written again unchanged, so this is safe and only slower. When the two
records were deleted and added back with the same ids, the run cannot tell
them from the originals and can skip the records added in between: run the
full `verify` after the last copy of a source that changed, and copy into
a new `--dir` when its ids check fails. A run that reads records again
gives the offset it reads from as `reread_from` in its JSON document; other
runs have no `reread_from`.

Each record that the tool cannot convert or that Weaviate refuses goes to
`rejects.jsonl` with the reason. Rejects are final for this directory. After
fixing the cause, copy `plan.yaml` and `inspect.json` to a new `--dir` and
run `copy` there; objects already in Weaviate are written again unchanged.
When Weaviate refuses every object of a batch of more than one record,
`copy` retries the batch as it retries a network error: it retries a
transient error for at least 15 s and at most about 30 s in total before it
gives up. When the retries are used up, it stops with `target_unreachable`
and stores no reject. The next run starts at that page. A record that
Weaviate refuses with an error about a node, a shard or a deadline, as it
does while a node restarts, is not a reject either: `copy` sends it again
under the same retry, and when the error stays, stops the same way.

When the target collection exists without a property that `plan.yaml`
has, `copy` adds the property before it writes. Its JSON document lists
each added property under `added_properties`.

A quantizer that Weaviate trains on the vectors (`sq`, `pq`, and `rq` with
`centering`) is enabled again after the last page of each plan entry. A
server without `ASYNC_INDEXING` then compresses the vectors at once. A
server with `ASYNC_INDEXING=true` compresses a shard only once it holds
`training_limit` vectors, which is 100000 for `sq` and `pq` and 10000 for
`rq` unless `plan.yaml` sets it, so a smaller shard stays uncompressed.
`check` and `copy` give the warning `quantizer_training` when the source
has fewer records than that limit. The entry is done only after that
update; a run that stops before it sends the update again and copies no
page again.

`state.sqlite` also records the target of each source collection: the
Weaviate host and port, the collection and the tenant. Once records of a
source collection are copied, `copy` stops with `state_mismatch` when
`WEAVIATE_URL` or the target in `plan.yaml` names another target, or when
the target collection no longer exists. To copy to another target, use
another `--dir`. Before any record is copied, a run takes the new target.

One `copy` at a time uses a directory. It holds a lock on `copy.lock` while
it runs, and a second `copy` on that directory exits with `usage`. The
operating system releases the lock when the process ends, so a stopped run
leaves no lock behind. On a system without `fcntl`, such as Windows, `copy`
runs without the lock.

- `--collection NAME`: copy only this plan entry. Repeatable.
- `--parallelism N`: reads per collection, 1 to 8. Default: 5.
- `--consistency ONE|QUORUM|ALL`: the write consistency level.

### `verify`

Compares Weaviate with Chroma: the counts, the full list of ids, and a
sample of records with their properties and vectors. It writes `report.md`
and `report.json`.

`copy` reads a live source by position. A record deleted from the source
during a copy can make `copy` skip other records. Run a full `verify` after
writes to the source have stopped. When the ids differ, the ids check names
the fix, as the count check does. `verify --quick` skips the comparison of
the full id lists. The counts can match when an extra object, such as one
whose source record was deleted, cancels a missing one; `--quick` then
finds the missing record only if the sample holds it. Run the full
`verify` after the last copy of a source that changed.

Weaviate stores an `int` property as a double, so an int above 2**53 in
magnitude is stored rounded, and `verify` compares the rounded value. An
int that rounds to 2**63 or above is a reject with the reason "int <v>
cannot be stored: Weaviate keeps ints as doubles and refuses this value;
type the property text", because whether the server takes such a value
depends on its CPU. To keep every digit of such a value, type the property
`text` in `plan.yaml`.

- `--sample N`: records to compare per collection, 1 to 10000. Default: 300.
- `--quick`: skip the comparison of the full id lists.

### `status`

Shows the copy progress and the recorded target of each collection from
`state.sqlite`. It needs no credentials. The total of a collection is the
highest source count that a `copy` read. The copied count is the number of
records that `copy` wrote, also those deleted from the source since, so
after deletions in the source it can be above the total. Records that a
run reads again are counted once.

## Migration directory

Every command takes `--dir`. The default is `./.to-weaviate`. The directory
holds `inspect.json`, `plan.yaml`, `state.sqlite`, `copy.lock`,
`events.jsonl`, `rejects.jsonl`, `report.md` and `report.json`.
When the disk of the directory is full, or the directory or a file in it
is read-only, a command stops with the error `directory_unwritable`, exit
2, naming the file and the cause. `copy` resumes at the page it was on
once the directory can be written.

## JSON output

Every command accepts `--json`. With it, stdout holds exactly one JSON
document, and progress lines and warnings go to stderr. Every document has
`status`, which is `success` or `error`. An error document also has `code`,
`message` and `hint`:

```json
{
  "status": "error",
  "code": "auth_failed",
  "message": "Weaviate refused the request: the API key is missing or wrong.",
  "hint": "Check WEAVIATE_API_KEY in the environment or in .env."
}
```

The exit code is the same with and without `--json`:

| Exit code | Meaning | Error codes |
|---|---|---|
| 0 | success | |
| 1 | rejected records or a failed verification | `rejects`, `verification_failed` |
| 2 | usage error, invalid plan, a state of another target, a state file that cannot be read, or a directory that cannot be written | `usage`, `invalid_plan`, `plan_mismatch`, `target_rejected_schema`, `state_mismatch`, `invalid_state`, `directory_unwritable` |
| 3 | missing credential or authentication failure | `missing_credential`, `auth_failed` |
| 4 | source or target unreachable | `source_unreachable`, `target_unreachable` |
| 5 | internal error | `internal_error` |

## Environment variables

The tool reads its settings from the environment and from `.env` in the
working directory. A value in the environment wins over `.env`. No option
takes a secret, and the tool prints no secret and writes none to a file.

| Variable | Use |
|---|---|
| `CHROMA_URL` | Chroma server. Default: `https://api.trychroma.com` (Chroma Cloud). |
| `CHROMA_API_KEY` | Chroma API key. Required for Chroma Cloud. |
| `CHROMA_TENANT` | Chroma tenant. Default: the tenant of the API key. |
| `CHROMA_DATABASE` | The database that `inspect chroma` reads when you give no `--database`. `--database` wins over it. With neither, `inspect chroma` reads every database of the tenant. |
| `CHROMA_TOKEN_HEADER` | Header that carries the Chroma key. Default: `x-chroma-token`. With `Authorization`, the tool sends `Bearer <key>`. |
| `WEAVIATE_URL` | Weaviate server. Required. |
| `WEAVIATE_API_KEY` | Weaviate API key. |
| `WEAVIATE_GRPC_URL` | Weaviate gRPC endpoint. Set it only for a server outside Weaviate Cloud. |
| `OPENAI_API_KEY`, `COHERE_API_KEY`, ... | Model provider keys. The tool sends them to Weaviate as headers for the vectorizer modules. `API_KEY_MAP` in `src/to_weaviate/settings.py` lists every variable; when two variables name one header, the first one in that list that is set wins. A provider that refuses the key during `copy` is the error `auth_failed`, which names the provider and its variables. |
| `TO_WEAVIATE_DEBUG` | `1` prints the traceback when a command fails. |

For a Weaviate server outside Weaviate Cloud, set both URLs, for example
`WEAVIATE_URL=http://localhost:8080` and
`WEAVIATE_GRPC_URL=http://localhost:50051`. Without `WEAVIATE_GRPC_URL`
the tool connects to port 443 of the `WEAVIATE_URL` host over TLS, so it
refuses a `WEAVIATE_URL` that names another port.

## Migrate from a local Chroma directory

You can serve a Chroma directory on disk, for example one that
`chromadb.PersistentClient(path=...)` wrote, with the `chroma` command from
the `chromadb` package:

```bash
pip install chromadb
chroma run --path /path/to/chroma-directory --port 8000
```

Then point the tool at that server. A server without authentication needs
no `CHROMA_API_KEY`:

```bash
export CHROMA_URL=http://localhost:8000
to-weaviate inspect chroma
```

## Development

The `test` extra does not install `pre-commit`, so install it next to it:

```bash
uv venv --python 3.10 .venv
uv pip install -e ".[test]" pre-commit
.venv/bin/python -m pytest tests/unit
.venv/bin/pre-commit run --all-files
```

### Acceptance suite

`tests/acceptance` runs the installed `to-weaviate` command against a real
Chroma server and a real Weaviate server. The tests start no server. They
run only against the test endpoints, so the four variables must have
exactly these values:

```bash
WEAVIATE_URL=http://localhost:8099
WEAVIATE_GRPC_URL=http://localhost:50099
WEAVIATE_API_KEY=admin-key
CHROMA_URL=http://localhost:8001
```

When a variable is missing or has another value, the suite skips every test
and names that variable. With `TO_WEAVIATE_REQUIRE_ACCEPTANCE=1`, as in CI,
the suite fails with the same text instead. It connects to no server before this check passes.
The commands it starts get empty values for `CHROMA_DATABASE`,
`CHROMA_TENANT`, `CHROMA_API_KEY` and `CHROMA_TOKEN_HEADER`, so values from
your shell do not reach them.

The suite has two levels. An API-level test reaches the servers only
through the `to-weaviate` command and the Chroma and Weaviate APIs. An
`infra` test also restarts or deletes a server through docker or kubectl.
Each `infra` test carries the pytest marker `infra`, which `pyproject.toml`
declares. `pytest -m "not infra"` runs the API-level tests only. With
`ACCEPTANCE_TARGET=cloud`, the suite runs against a Weaviate Cloud cluster
and the local test Chroma. `endpoint_problems` in `tests/e2e.py` holds the
rules of both targets. On the cloud target, an `infra` test fails before
its fixtures run, because docker and kubectl reach only the local servers.
The cloud target must be a serverless cluster. A sandbox cluster refuses
`hnsw`, `flat`, `dynamic` and the quantizer settings, so those cases fail
there with the server's message.

The suite deletes and recreates the Chroma collections that `COLLECTIONS`
in `tests/acceptance/chroma/seed.py` lists. One test adds 10 records to
`tiny.l2` and deletes them when it ends. Before and after each test the
suite deletes the Weaviate collections that `SUITE_COLLECTIONS` in
`tests/acceptance/conftest.py` lists. It writes to no other Weaviate
collection, also when the test Chroma holds collections that the seed does
not make. With `ACCEPTANCE_TARGET=cloud`, the suite fails before its first
test when the cluster holds a collection that `SUITE_COLLECTIONS` does not
list, so the target must be a dedicated cluster.

CI runs the suite in two legs, each on its own kind cluster. Both run
Weaviate with `ASYNC_INDEXING`, as every Weaviate Cloud instance does, and
set `WEAVIATE_ASYNC_INDEXING=true` for the tests whose outcome depends on
it.
The `single` leg runs one Weaviate node. The `cluster` leg runs three
nodes with `REPLICATION_MINIMUM_FACTOR=3` and sets `ACCEPTANCE_NODES=3`.
The cases of `tests/acceptance/chroma/test_cluster.py` run there only: they
delete a Weaviate pod during `copy` and during `verify`, and copy at each
consistency level. They skip when `ACCEPTANCE_NODES` is below 3.
`tests/acceptance/chroma/test_chroma_restart.py` stops the compose Chroma
for 3 s during a copy, on both legs and locally. Right after the
weaviate-local-k8s action, the job runs `tests/acceptance/port-forward.sh`.
It replaces the action's `kubectl-relay` forwards on ports 8099 and 50099
with `kubectl port-forward` processes. A loop starts each of them again
when it exits. The script then waits until Weaviate answers on both
ports. It refuses to run when `GITHUB_ACTIONS` is not `true`.

With `ASYNC_INDEXING`, the copy of a dynamic index passes, and the
compression of the tenants by a trained quantizer with the default
`training_limit` is an expected failure. Without it, the dynamic index is
an expected failure with the server's message as the reason; it describes
a finding that is not fixed. The compression of the tenants passes there.
CI does not run that server. Unit tests cover the tool's side. The suite
runs against such a server locally with `ASYNC=0 make weaviate-up`, then
`WEAVIATE_ASYNC_INDEXING=false make acceptance`. With
`WEAVIATE_ASYNC_INDEXING` unset, these tests decide from what the server
does and say so in the reason.

The `Makefile` starts the servers and sets the variables. `make` alone
prints the targets. `make acceptance-up` and `make acceptance-down` call
`tests/acceptance/cluster.sh up` and `down`. `make acceptance-up` needs a
clone of [weaviate-local-k8s](https://github.com/weaviate/weaviate-local-k8s)
at `LOCAL_K8S` (default: `../weaviate-local-k8s`), Docker, and every program
that `local-k8s.sh` uses: `kind`, `helm`, `kubectl`, `curl`, `nohup`, `jq` and
`lsof`, plus `gtimeout` on macOS (from coreutils) or `timeout` on Linux.

```bash
make acceptance-up     # Chroma on port 8001, Weaviate on ports 8099 and 50099 in a kind cluster
source .venv/bin/activate
make acceptance        # pytest tests/acceptance -v
make acceptance-down   # stops Chroma, then asks before it deletes the kind cluster
```

`make acceptance-up` creates a kind cluster named `weaviate-k8s` and
switches the kubectl context to `kind-weaviate-k8s`. Before it starts
anything, it checks that `kind` is on `PATH`, that
`$LOCAL_K8S/local-k8s.sh` is executable, and that `kind get clusters`
succeeds and lists no cluster named `weaviate-k8s`, because
`local-k8s.sh setup` would reuse that name. After `setup` succeeds, it
writes the id of the container `weaviate-k8s-control-plane` to
`.acceptance-kind-cluster`, which git ignores. When `setup` fails, it writes
no marker, so `make acceptance-down` will not remove a partly created
cluster. It then prints the command that removes the cluster by hand:
`kind delete cluster --name weaviate-k8s`.

`make acceptance-down` always stops Chroma. It deletes the cluster only when
`.acceptance-kind-cluster` exists and holds the id of the
`weaviate-k8s-control-plane` container that exists now. When that container
is not found or has another id, it removes the stale marker and leaves the
cluster alone. Otherwise it lists every effect of `local-k8s.sh clean` and
runs it only after you type `y` on a terminal. With stdin from a pipe or a
file, it does not run `clean` and prints the command to run by hand. `clean` deletes the kind cluster
`weaviate-k8s`, switches the kubectl context to `kind-weaviate-k8s` first,
and kills every `kubectl-relay` process and every
`kubectl get pods -n weaviate -w` process on the machine. After it, no
kubectl context is current, so select your previous context again with
`kubectl config use-context NAME`.

`make acceptance` works with any servers on those ports: Chroma from
`tests/acceptance/chroma/docker-compose.yml`, and a Weaviate server that
accepts the API key `admin-key`. The Weaviate version of CI is in
`tests/acceptance/weaviate-version`.

#### Run the suite locally with Docker

`make acceptance-up` stops when the machine already has a kind cluster
named `weaviate-k8s`. On such a machine, or on one without kind, start
the servers in Docker instead:

```bash
make chroma-up       # Chroma on port 8001
make weaviate-up     # Weaviate and text2vec-transformers on ports 8099 and 50099
source .venv/bin/activate
WEAVIATE_ASYNC_INDEXING=true make acceptance
make weaviate-down   # removes the two containers and their network
make chroma-down     # stops Chroma
```

`make chroma-up` and `make chroma-down` run `docker compose` with
`tests/acceptance/chroma/docker-compose.yml`. `make weaviate-up` and
`make weaviate-down` run `tests/acceptance/weaviate-docker.sh`.

`make weaviate-up` starts two containers on the network
`to-weaviate-dev-net`. `to-weaviate-dev` runs the Weaviate version in
`tests/acceptance/weaviate-version`. `to-weaviate-t2v` serves the
`text2vec-transformers` model of CI. Weaviate listens on `127.0.0.1`
only. It accepts the API key `admin-key` and has RBAC on with the root
user `admin-user`. The script asks `/v1/.well-known/ready` up to 60
times, 2 seconds apart. When Docker does not answer, it stops with a
message. When one of the two containers or the network
exists, it starts nothing and asks for `make weaviate-down` first.

Weaviate runs with `ASYNC_INDEXING=true`, as every Weaviate Cloud
instance does. `ASYNC=0 make weaviate-up` starts it without async
indexing. Against that server, run the suite with
`WEAVIATE_ASYNC_INDEXING=false make acceptance`.

`make weaviate-down` removes the two containers and the network. It
skips a name that does not exist.

#### Run the API-level suite against a Weaviate Cloud cluster

`make acceptance-cloud` runs the API-level tests against a Weaviate Cloud
cluster. The source stays the local test Chroma, so start it first with
`make chroma-up`. The target runs `pytest tests/acceptance -v -m "not infra"`
with these variables:

| Variable | Value |
|---|---|
| `ACCEPTANCE_TARGET` | `cloud` |
| `CHROMA_URL` | `http://localhost:8001` |
| `WEAVIATE_ASYNC_INDEXING` | `true` |
| `ACCEPTANCE_VECTORIZER` | `text2vec-weaviate` |
| `WEAVIATE_GRPC_URL` | empty, so the tool connects to port 443 of the cluster |
| `TO_WEAVIATE_REQUIRE_ACCEPTANCE` | `1`, so a target that breaks a rule fails the run |

It reads `WEAVIATE_URL` and `WEAVIATE_API_KEY` from the environment.
`WEAVIATE_URL` is `https://` followed by the cluster host, which ends with
`.weaviate.cloud` or `.weaviate.network`. `WEAVIATE_API_KEY` is an admin
key of the cluster. Put the two `export` lines in a shell file outside the
repository and `source` it:

```bash
make chroma-up
source ~/weaviate-cloud-test.sh   # export WEAVIATE_URL=... and export WEAVIATE_API_KEY=...
source .venv/bin/activate
make acceptance-cloud
```

A value typed at the prompt, in an `export` or on the make command line,
stays in the shell history. Never set them on the make command line:
make also passes such a value to every child process in `MAKEFLAGS`. When
either variable is empty or set on the make command line, the target
prints `set WEAVIATE_URL and WEAVIATE_API_KEY in the environment (not on
the make command line)` and exits 2. It prints no value.

Use a dedicated serverless cluster. Before and after each test, the suite
deletes the collections that `SUITE_COLLECTIONS` in
`tests/acceptance/conftest.py` lists. It fails before its first test when
the cluster holds a collection that `SUITE_COLLECTIONS` does not list. A
sandbox cluster refuses `hnsw`, `flat`, `dynamic` and the quantizer
settings, so those cases fail there with the server's message.
`/v1/meta` of the cluster must list `text2vec-weaviate`. When it does not,
the tests that embed documents and the module check of `test_check.py`
fail.

Every API-level test runs. The target deselects the `infra` tests,
because docker and kubectl reach only the local servers. The
compression of the tenants by a trained quantizer is an expected failure,
as on every server with `ASYNC_INDEXING`. When `WEAVIATE_URL` does not use
`https` or its host has another suffix, every test fails before it starts.
The failure text names the broken rule, for example `WEAVIATE_URL does not
use https (the cloud target needs TLS)`. The target then exits non-zero.

### Releases

`.github/workflows/release.yml` runs when a tag that matches
`v[0-9]+.[0-9]+.[0-9]+*` is pushed. The `build` job stops unless the tag
points at a commit on `main` and the built wheel has the tag's version
without the `v`. The `publish` job uploads the wheel and the sdist to PyPI
through trusted publishing. It uses no token. The workflow runs no tests,
because no Chroma Cloud account exists for a test job.

Before the tag, the owner checks two runs of the commit to tag. The
release notes record the result of each: passed, skipped with the
reasons, failed.

- `make acceptance-cloud` against a dedicated serverless Weaviate Cloud
  cluster, from a laptop.
- The full suite in CI: the `CI` workflow of that commit on `main`.

A person with access to the GitHub repository and to PyPI sets up two
things once:

- PyPI has a pending trusted publisher for the project `to-weaviate`:
  repository `weaviate/to-weaviate`, workflow `release.yml`, environment
  `pypi`.
- The GitHub environment `pypi` allows deployments from tags that match
  `v*`. GitHub rejects the `publish` job of a tag that the environment
  does not allow.

A release is a tag on a commit of `main`. The `build` job fails for a tag
on any other commit:

```bash
git fetch origin
git tag v0.1.0 origin/main
git push origin v0.1.0
```

When the two jobs pass, `uvx to-weaviate --version` prints
`to-weaviate 0.1.0`.
