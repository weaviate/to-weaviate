# Architecture

`to-weaviate` is a Python package with a [Typer](https://typer.tiangolo.com)
command-line interface. The entry point is `to_weaviate.cli:app`.

## Layers

```mermaid
flowchart TB
    cli["cli.py: commands, settings, errors"]
    engine["engine/: check, copy, verify"]
    sources["sources/: the Source protocol and one package per source"]
    target["target/: Weaviate schema, writer, reader"]
    dir["migration directory: plan.py, state.py, events.py, lock.py"]
    cli --> engine
    cli --> dir
    engine --> sources
    engine --> target
    engine --> dir
```

- `cli.py` parses a command, reads the settings, builds the source and the
  Weaviate clients, and maps every exception to an error code.
- `engine/` runs check, copy and verify through the `Source` protocol and
  imports no source package.
- `sources/` reads a source database. `sources/base.py` defines the protocol;
  `sources/chroma/` implements it.
- `target/` talks to Weaviate through the
  [Weaviate Python client](https://github.com/weaviate/weaviate-python-client).
- The migration directory holds the plan, the copy state, the events and the
  lock.

## Modules

| Module | Responsibility |
|---|---|
| `cli.py` | The commands; maps exceptions to `CliError`; loads `plan.yaml` and `inspect.json`; checks the state and the target before a copy |
| `settings.py` | `load_env` (`.env` under the environment), the source and Weaviate settings, URL validation, `API_KEY_MAP` |
| `output.py` | `ERROR_EXIT` (error code to exit code), JSON and text output, secret redaction |
| `plan.py` | The plan model, `draft_plan`, `load_plan`, `save_plan`, and the index and quantizer rules |
| `state.py` | `state.sqlite`: read positions, rejects, recorded targets, quantizer updates |
| `events.py` | `events.jsonl` |
| `lock.py` | `copy.lock` |
| `retry.py` | `with_retry`: 7 attempts, random delays up to 30 s |
| `naming.py` | Weaviate's rules for collection, property and tenant names |
| `types.py` | Metadata value to Weaviate type, and the conversion of values |
| `ids.py` | `to_uuid`: source id to object UUID |
| `engine/check.py` | Plan findings: names, types, modules, provider keys, existing collections, tenants, quantizer training |
| `engine/copy.py` | `copy_collection`: the partitioned, threaded, resumable copy and the tail of new records |
| `engine/verify.py` | The `count`, `ids` and `records` checks, and the report |
| `sources/base.py` | The `Source` protocol and the types `SourceCollection`, `FieldStats`, `Record`, `Partition`, `Page` |
| `target/connect.py` | Weaviate Cloud or custom connection, from the settings |
| `target/schema.py` | Creates collections from the plan, compares an existing collection with it, updates quantizers, classifies transient errors |
| `target/writer.py` | Record to Weaviate object or reject; batches of at most 4 MiB with retries |
| `target/reader.py` | Reads objects back for `verify` |

## How each command runs

**`inspect <source>`**: `cli.py` builds the source from its settings and
calls `Source.inspect`, which returns one `SourceCollection` per collection.
`plan.draft_plan` turns them into a `Plan`, with the source's own mapper from
embedding configuration to vectorizer. `cli.py` writes `inspect.json` and
`plan.yaml`.

**`check`**: `cli.py` loads the plan. `engine/check.py` returns findings from
the plan alone and from the modules the target has enabled. `cli.py` adds the
provider keys from the environment and the differences that
`target/schema.py` finds in existing collections. `check` writes nothing.

**`copy`**:

1. `cli.py` runs the plan findings of `check`, the source check and the
   provider-key check. It then takes `copy.lock` and checks the enabled
   modules and the recorded targets in `state.sqlite`.
2. `target/schema.ensure_collection` creates each target collection, or
   compares the existing one with the plan and adds missing properties.
3. For each plan entry, `engine/copy.copy_collection`:
   - calls `Source.wait_compacted`, which returns a state, or `None` for a
     source without compaction;
   - splits the collection with `Source.partitions` and reads each partition
     in its own thread with `Source.read`;
   - converts each page with `target/writer` and writes it, then advances
     the partition's position in `state.sqlite`;
   - reads `Source.count` again and copies the records added since in a tail
     partition;
   - compares `Source.compaction_state` with the state from the wait, and
     raises `SourceChanged` when they differ.
4. `copy` enables the trained quantizers of each finished entry.

**`verify`**: `engine/verify.py` compares `Source.count` with the object
count, diffs `Source.ids` against the target ids in a scratch sqlite file,
and compares `Source.sample` records with the objects that `target/reader`
reads. It writes `report.md` and `report.json`.

**`status`**: reads `plan.yaml` and `state.sqlite`.

## The `Source` protocol

`Source` in `sources/base.py` is a `typing.Protocol`. mypy checks that a
source conforms; nothing checks it at run time.

| Member | Called by |
|---|---|
| `inspect(namespaces, collections, sample_pages)` | `inspect`. The result goes to `inspect.json` and into the draft plan. |
| `partitions(collection, parallelism)` | `copy`, once per collection. Stored in `state.sqlite`. |
| `read(collection, partition, cursor, with_vectors)` | `copy`, one thread per partition. Returns a `Page`; `next_cursor` `None` ends the partition. |
| `ids_at(collection, offset, limit)`, `page_size` | `copy`, to find its place in the tail after deletions. |
| `count(collection)` | `copy` (the live total and the tail) and `verify` (the `count` check). |
| `ids(collection)` | `verify`, the `ids` check. |
| `sample(collection, size, rng, with_vectors)` | `verify`, the `records` check. |
| `wait_compacted(collection, waiting)`, `compaction_state(collection)` | `copy` and `verify`, before the first page and after the last read. |
| `tenant`, `searched_namespaces` | `inspect`, for the message when it finds no collection. |
| `namespaces()` | The source's own `inspect`, when no namespace is given. |
| `name`, `capabilities`, `vector_search`, `filter_equal` | Declared by the protocol; no command calls them. |

### What the copy engine expects of a source

| Rule | Why |
|---|---|
| `partitions` returns ranges of decimal offsets that cover `[0, count)`, each with a `stop`, and `[]` for an empty collection. | `engine/copy.py` parses the offsets to know what is copied. |
| Cursors are decimal offsets. A tail partition (`stop` is `None`) reads until an empty page. | The tail starts where the last run ended. |
| Records come in insertion order. A deletion moves later records one position down; an addition goes after the last record. | Resuming and finding new records depend on this order. |
| `read` is safe to call from up to 8 threads for different partitions of one collection. | `copy --parallelism` runs one thread per partition. |
| A vector that holds a non-number arrives as NaN. | The engine turns such a record into a reject. |
| `count` equals the number of records `read` returns. | `verify`'s `count` check compares it with Weaviate. |
| `inspect` lists every metadata key in `fields`. | A key missing from the plan makes each record that has it a reject. |
| Metadata values are `bool`, `int`, `float`, `str`, `None`, or lists of one of these. | `types.py` maps and converts only these. |
| Errors are the source's own exception class, with a `status` and an optional `hint`. | `cli.py` maps them to `auth_failed` or `source_unreachable`. |
| Every `SourceCollection` field survives `dataclasses.asdict` to JSON and back. | `inspect.json` stores it. |

A source that pages with an opaque cursor, such as a token or the last id,
breaks the offset rules. Supporting one means changing the resume and tail
logic in `engine/copy.py`.

## Rules every change keeps

- No command writes to the source.
- A read position advances only after Weaviate stored its page.
- An object's UUID depends only on its source id (`ids.py`), so writing a
  page twice adds nothing.
- Secrets come only from the environment or `.env`. Output and files redact
  the values of `SECRET_VARIABLES` (`output.py`) and the passwords and tokens
  in `URL_VARIABLES` (`settings.py`). A new secret variable goes into one of
  those lists.
- With `--json`, stdout holds exactly one JSON document.
- Every error code has one exit code, in `ERROR_EXIT` (`output.py`).
