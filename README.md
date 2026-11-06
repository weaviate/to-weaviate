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
  Weaviate vectorizer embed the documents, or both.
- `--vectorizer MODULE`, `--vectorizer-option KEY=VALUE`: the Weaviate
  vectorizer module and its options.
- `--index`, `--quantizer`, `--quantizer-bits`: the vector index type, the
  quantizer and the bits for `rq` (1, 4 or 8). `default` leaves the choice
  to the server.
- `--force`: replace an existing `plan.yaml`.

### `check`

Checks `plan.yaml` against the Weaviate server: the plan itself, the
vectorizer modules, and existing collections with other settings. It
creates nothing.

### `copy`

Creates the target collections, then copies the records. Progress goes to
`state.sqlite`, so a second run continues where the first one stopped.
Each record that the tool cannot convert or that Weaviate refuses goes to
`rejects.jsonl` with the reason.

- `--collection NAME`: copy only this plan entry. Repeatable.
- `--parallelism N`: reads per collection, 1 to 8. Default: 5.
- `--consistency ONE|QUORUM|ALL`: the write consistency level.

### `verify`

Compares Weaviate with Chroma: the counts, the full list of ids, and a
sample of records with their properties and vectors. It writes `report.md`
and `report.json`.

- `--sample N`: records to compare per collection, 1 to 10000. Default: 300.
- `--quick`: skip the comparison of the full id lists.

### `status`

Shows the copy progress from `state.sqlite`. It needs no credentials.

## Migration directory

Every command takes `--dir`. The default is `./.to-weaviate`. The directory
holds `inspect.json`, `plan.yaml`, `state.sqlite`, `events.jsonl`,
`rejects.jsonl`, `report.md` and `report.json`.

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
| 2 | usage error or invalid plan | `usage`, `invalid_plan`, `plan_mismatch`, `target_rejected_schema` |
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
| `OPENAI_API_KEY`, `COHERE_API_KEY`, ... | Model provider keys. The tool sends them to Weaviate as headers for the vectorizer modules. `API_KEY_MAP` in `src/to_weaviate/settings.py` lists every variable. |
| `TO_WEAVIATE_DEBUG` | `1` prints the traceback when a command fails. |

For a Weaviate server outside Weaviate Cloud, set both URLs, for example
`WEAVIATE_URL=http://localhost:8080` and
`WEAVIATE_GRPC_URL=http://localhost:50051`.

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

The suite deletes and recreates the Chroma collections `docs-cosine`,
`tiny.l2` and `docs-rq`. Before each test it deletes the Weaviate
collections `Docs_cosine`, `Tiny_l2` and `Docs_rq`.

The `Makefile` starts the servers and sets the variables. `make` alone
prints the targets. `make acceptance-up` needs a clone of
[weaviate-local-k8s](https://github.com/weaviate/weaviate-local-k8s) at
`LOCAL_K8S` (default: `../weaviate-local-k8s`), Docker, and every program
that `local-k8s.sh` requires: `kind`, `helm`, `kubectl`, `curl`, `nohup` and
`jq`, plus `gtimeout` on macOS (from coreutils) or `timeout` on Linux.

```bash
make acceptance-up     # Chroma on port 8001, Weaviate on ports 8099 and 50099 in a kind cluster
source .venv/bin/activate
make acceptance        # pytest tests/acceptance -v
make acceptance-down   # stops Chroma, then asks before it deletes the kind cluster
```

`make acceptance-up` creates a kind cluster named `weaviate-k8s` and
switches the kubectl context to `kind-weaviate-k8s`. Before it starts
anything, it checks that `kind` is on `PATH` and that
`$LOCAL_K8S/local-k8s.sh` is executable. It stops with an error when a kind
cluster named `weaviate-k8s` already exists, because `local-k8s.sh setup`
would reuse that name. Before `setup` runs, it writes the file
`.acceptance-kind-cluster`, which git ignores.

`make acceptance-down` always stops Chroma. It deletes the cluster only when
`.acceptance-kind-cluster` exists, so it leaves alone a cluster that this
checkout did not create. It lists every effect of `local-k8s.sh clean` and
runs it only after you type `y`. `clean` deletes the kind cluster
`weaviate-k8s`, switches the kubectl context to `kind-weaviate-k8s`, and
kills every `kubectl-relay` process and every
`kubectl get pods -n weaviate -w` process on the machine.

`make acceptance` works with any servers on those ports: Chroma from
`tests/acceptance/chroma/docker-compose.yml`, and a Weaviate server that
accepts the API key `admin-key`. The Weaviate version of CI is in
`tests/acceptance/weaviate-version`.
