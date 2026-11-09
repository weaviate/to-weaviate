# Adding a source

A new source is a package under `src/to_weaviate/sources/<name>/` that
implements the [`Source` protocol](architecture.md#the-source-protocol), plus
edits in `cli.py` and `settings.py`. The copy engine and the Weaviate side
need no change for a source that reads by offset in insertion order. Read
[what the copy engine expects of a source](architecture.md#what-the-copy-engine-expects-of-a-source)
first.

Today only Chroma exists, so a few shared places name Chroma. The last
section lists them.

## Steps

1. **Client.** Create `sources/<name>/client.py`: the transport, the
   source's exception class with a `status` and an optional `hint`, a
   transient-error check for `with_retry` (`retry.py`), and redaction of
   secrets from error text. `sources/chroma/client.py` is the model.
2. **Source.** Create `sources/<name>/source.py` with every member of the
   protocol. Add `config.py` for the distance and embedding configuration,
   and `mapping.py` for embedding configuration to Weaviate vectorizer, if
   the source stores them. The mapping returns a secret-free form for
   `inspect.json`.
3. **Settings.** Add a settings dataclass and a reader next to
   `ChromaSettings` and `chroma_settings` in `settings.py`, with the
   `<NAME>_*` variables. Add URL variables to `URL_VARIABLES` and key
   variables to `SECRET_VARIABLES` in `output.py`. Values of variables in
   neither list are not redacted.
4. **Inspect command.** Add `to-weaviate inspect <name>` next to
   `inspect chroma` in `build_app` (`cli.py`). It takes the source's own
   options to select collections and the shared vector options
   (`--vectors`, `--vectorizer`, `--vectorizer-option`, `--index`,
   `--quantizer`, `--quantizer-bits`, `--force`). It writes `type: <name>` in
   the plan's `source` block.
5. **Source selection.** Make `copy` and `verify` build the source from the
   plan's `source.type`. See [where the code names Chroma](#where-the-code-names-chroma).
6. **Errors.** Map the new exception class to `auth_failed` and
   `source_unreachable` in `_as_cli_error` (`cli.py`), and write hints that
   name the new variables.
7. **Unit tests.** Mirror the Chroma modules: `test_<name>_client.py`,
   `test_<name>_source.py`, `test_<name>_mapping.py`, and `test_cli_<name>.py`
   with a fake server behind `httpx.MockTransport` (see
   `tests/unit/chroma_server.py`). Add the new variables to
   `test_settings.py` and `test_output.py`, and the new errors to the error
   table of `test_cli_json.py`. `test_copy.py` runs over `FakeSource` and
   needs no copy.
8. **Acceptance tests.** Create `tests/acceptance/<name>/` with a `seed.py`,
   a compose file or a start script for the test server, and test modules
   that start with `inspect <name>`. Extend the shared rules:
   `LOCAL_ENDPOINTS` and the cloud rule in `tests/e2e.py`,
   `CLEARED_VARIABLES` and `SUITE_COLLECTIONS` in
   `tests/acceptance/conftest.py`, and `<name>-up` and `<name>-down` targets
   in the `Makefile`. See [Testing](testing.md).
9. **CI.** The `unit` and `acceptance` jobs of `.github/workflows/ci.yml` run
   every module under `tests/unit` and `tests/acceptance`. Add a step that
   starts the new test server and put its URL in the job environment.
10. **Docs.** Copy [`docs/sources/_template.md`](../sources/_template.md) to
    `docs/sources/<name>.md` and fill every section. Add a row to the sources
    tables in `README.md` and `docs/sources/README.md`.

## Where the code names Chroma

These places assume the only source is Chroma. A second source changes them
instead of only adding code.

| Place | Today | What a second source needs |
|---|---|---|
| Source selection in `cli.py` | `copy` and `verify` call `chroma_settings` and build a `ChromaSource`. `SourceFactory` takes `ChromaSettings`. | A dispatch on the plan's `source.type`, which `inspect` writes and nothing reads yet. |
| The embedding mapper | `cli.py` passes the Chroma `vectorizer_for` to `draft_plan` and to the plan findings of `check` and `copy`. | The mapper of the plan's source. |
| Error mapping | `_as_cli_error` knows `ChromaError`; a raw `httpx.TransportError` becomes `target_unreachable`. | The new exception class. |
| Names in shared models and messages | `SourceRef.namespace` and `SourcePlan.tenant` use Chroma's database and tenant. Hints name `inspect chroma` and `CHROMA_*`. | Neutral wording, or the source's own words. |
| Sparse vectors and URIs | `types.value_type` knows Chroma's sparse-vector shape; the plan drafts a `uri` property for every collection. | A translation in the source's `read`, or a change in `types.py`. |
| Unit test helpers | `FakeSource` in `tests/unit/fakes.py` mirrors Chroma's offset partitions; `FakeFactories` takes `ChromaSettings`. | Helpers that take any settings. |
| Offset paging in the copy engine | `engine/copy.py` treats cursors as decimal offsets and relies on insertion order. | A source with opaque cursors needs a new resume and tail design. |
