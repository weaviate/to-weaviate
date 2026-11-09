# Commands

This page describes the commands that every source shares. The options that
select what to read from a source are on the source's guide, under
[Sources](sources/README.md).

## Settings and common options

The tool reads its settings from environment variables and from a `.env` file
in the working directory. A value in the environment wins over `.env`. No
option takes a secret.

- Source settings: the guide of each source, for example
  [Chroma](sources/chroma.md#connection-settings).
- Target settings: [Weaviate target](weaviate.md).
- `TO_WEAVIATE_DEBUG=1` prints the traceback when a command fails.

Every command takes these options:

| Option | Effect |
|---|---|
| `--dir PATH` | The [migration directory](#migration-directory). Default: `./.to-weaviate`. |
| `--json` | Print one JSON document on stdout. Progress lines and warnings go to stderr. See [JSON output](#json-output). |

`to-weaviate --version` prints the version.

## `inspect <source>`

Each source has its own inspect command, such as `inspect chroma`. It reads
the source collections and writes two files:

- `inspect.json`: what the source holds. For each collection: the record
  count, the vector dimension and distance, the embedding configuration, and
  every metadata key with the types of its values.
- `plan.yaml`: a draft of the migration. Review and edit it before you run
  `check`. [The plan file](plan-file.md) describes its fields.

`inspect` refuses to replace an existing `plan.yaml` unless you pass
`--force`. It writes nothing to Weaviate.

Every inspect command takes these options. They set the vector settings of
the draft plan:

| Option | Effect |
|---|---|
| `--vectors keep\|embed\|both` | `keep` (default) copies the source vectors. `embed` lets a Weaviate vectorizer embed the documents. `both` writes the source vectors and the embedded ones as two named vectors. See [vector modes](plan-file.md#vector-modes). |
| `--vectorizer MODULE` | The Weaviate vectorizer module, for example `text2vec-openai`. With `embed` and no module, the plan uses `text2vec-weaviate`. |
| `--vectorizer-option KEY=VALUE` | An option of the vectorizer module. Repeatable. `plan.yaml` holds each value as text, and `copy` sends it as a number or a boolean where the module takes one. |
| `--index TYPE` | `default`, `hnsw`, `flat`, `dynamic` or `hfresh`. With `default`, the server chooses. |
| `--quantizer TYPE` | `default`, `none`, `rq`, `sq`, `pq` or `bq`. |
| `--quantizer-bits N` | The bits of `rq`: 1, 4 or 8. |
| `--force` | Replace an existing `plan.yaml`. |

Weaviate accepts only some index and quantizer combinations.
[Index and quantizer](plan-file.md#index-and-quantizer) lists them;
`inspect`, `check` and `copy` refuse the others.

## `check`

`check` compares `plan.yaml` with the Weaviate server and creates nothing.
It prints each finding as an error or a warning. An error stops `check` with
exit 2, `invalid_plan`. A warning does not stop `copy`.

`check` finds, among others:

- Names that Weaviate refuses (`invalid_name`). A tenant name has 1 to 64
  letters, digits, `_` or `-`.
- A vectorizer module that the server has not enabled (`module_not_enabled`).
- A missing provider key for a vectorizer: exit 3, `missing_credential`.
- An existing collection whose settings differ from the plan
  (`plan_mismatch`).
- Plan entries that write into one collection, each into its own tenant,
  but differ in their vectors, in the settings of a shared property, or in
  two property names that differ only in case.
- Warnings: `missing_property` (an existing collection lacks a planned
  property; `copy` adds it), `coerced_keys` (a metadata key with mixed types
  whose values `copy` converts), and `quantizer_training` (a trained
  quantizer whose training limit is above the record count of the source).

## `copy`

`copy` creates the target collections, then copies the records page by page.

| Option | Effect |
|---|---|
| `--collection NAME` | Copy only this plan entry. It matches the source or the target name. Repeatable. |
| `--parallelism N` | Reads per collection at the same time, 1 to 8. Default: 5. |
| `--consistency ONE\|QUORUM\|ALL` | The write consistency level. |

### Resuming

`copy` records its progress in `state.sqlite`. A read position advances only
after Weaviate has stored the page. When a run stops for any reason, the next
`copy` in the same directory starts at the page that was not stored. Each
object's UUID comes from its source id, so a page written twice replaces the
same objects and adds none.

One `copy` at a time can use a directory. While it runs it holds a lock on
`copy.lock`, and a second `copy` exits with `usage`. The operating system
releases the lock when the process ends. On a system without `fcntl`, such as
Windows, `copy` runs without the lock.

### Records added or deleted between runs

Each run also copies the records added to the source since the last run. It
finds its place from the last two records the previous run read:

- When deletions in the source moved those records, the run finds them at
  their new position and reads what follows.
- When one of them is gone, the run reads again from an earlier record or
  from the start, with the warning "the source changed since the last run;
  re-reading from the start". Records read again are written again unchanged.
  The JSON document of such a run gives the position it read from as
  `reread_from`.
- When the two records were deleted and added back with the same ids, the run
  cannot tell them apart from the originals and can skip the records added in
  between. Run the full `verify` after the last copy of a source that changed.

### Rejects

`copy` writes each record that it cannot convert, or that Weaviate refuses,
to `rejects.jsonl` with the reason. It then exits 1, `rejects`. A reject is
final for its directory. After you fix the cause, copy `plan.yaml` and
`inspect.json` into a new `--dir` and run `copy` there. Objects already in
Weaviate are written again unchanged.

### Retries

`copy` retries a transient error: a network error, a timeout, HTTP 429 or
5xx, and the matching gRPC statuses. It tries up to 7 times, for 15.75 s to
31.5 s in total. It also retries:

- a batch whose every object Weaviate refused;
- an object that Weaviate refused with an error about a node, a shard or a
  deadline, as it does while a node restarts.

When the retries run out, `copy` stops with `source_unreachable` or
`target_unreachable`, exit 4, and stores no reject. The next run starts at
that page. A refused API key is not retried: `copy` stops with
`auth_failed`, exit 3.

### Existing collections and the recorded target

When a target collection exists, `copy` compares it with the plan. It adds a
planned property that the collection lacks and lists it under
`added_properties` in its JSON document. Any other difference stops it with
`plan_mismatch`, exit 2.

`state.sqlite` records the target of each source collection: the Weaviate
host and port, the collection and the tenant. Once records of a collection
are copied, `copy` stops with `state_mismatch`, exit 2, when `WEAVIATE_URL`
or the plan names another target, or when the target collection is gone. To
copy to another target, use another `--dir`.

`copy` and `verify` also stop with `invalid_plan` before they connect when
the source settings name another source than the one `inspect` recorded in
`plan.yaml`.

## `verify`

`verify` compares each target collection with its source and writes
`report.md` and `report.json`. It runs three checks:

| Check | What it compares |
|---|---|
| `count` | The record count of the source and the object count of Weaviate. |
| `ids` | The full list of ids on both sides. It names the missing and the extra ones. |
| `records` | A random sample of records: every property, and the vectors within 1e-5. |

A failed check exits 1, `verification_failed`, and its detail names the fix.

| Option | Effect |
|---|---|
| `--sample N` | Records to compare per collection, 1 to 10000. Default: 300. |
| `--quick` | Skip the `ids` check. |

Run the full `verify` after writes to the source have stopped. A record
deleted from the source during a copy can make `copy` skip another record.
The counts can match when an extra object cancels a missing one; `--quick`
then finds the missing record only if the sample holds it.

## `status`

`status` shows, for each collection, the copied count, the total, the
rejects, whether the copy is done, and the recorded target. It reads only
`state.sqlite` and needs no credentials, so you can run it during a copy.

The total is the highest source count that a `copy` read. The copied count
includes records deleted from the source since, so after deletions it can be
above the total.

## Migration directory

All files of a migration live in one directory, `./.to-weaviate` unless you
pass `--dir`:

| File | Written by | Holds |
|---|---|---|
| `inspect.json` | `inspect` | What the source holds |
| `plan.yaml` | `inspect`, then you | The migration plan |
| `state.sqlite` | `copy` | Progress, read positions and the recorded targets |
| `copy.lock` | `copy` | The lock of the running `copy` |
| `events.jsonl` | `copy` | Progress events, at most one every 5 s per collection |
| `rejects.jsonl` | `copy` | One line per rejected record, with the reason |
| `report.md`, `report.json` | `verify` | The verification report |

When the disk is full, or the directory or a file in it is read-only, a
command stops with `directory_unwritable`, exit 2, and names the file and
the cause. `copy` resumes at its page once the directory can be written.

## JSON output

With `--json`, stdout holds exactly one JSON document and every document has
`status`: `success` or `error`. An error document also has `code`, `message`
and `hint`:

```json
{
  "status": "error",
  "code": "auth_failed",
  "message": "Weaviate refused the request: the API key is missing or wrong.",
  "hint": "Check WEAVIATE_API_KEY in the environment or in .env."
}
```

A successful `copy`:

```json
{
  "status": "success",
  "collections": [
    {
      "ref": "451fdd2c-b3ba-4224-8117-97d6a271e002",
      "target": "Docs_cosine",
      "total": 700,
      "copied": 700,
      "rejects": 0,
      "done": true
    }
  ],
  "total": 700,
  "added_properties": [],
  "warnings": [
    {
      "code": "coerced_keys",
      "collection": "Docs_cosine",
      "message": "key code has the types int: 650, text: 50; values are stored as text"
    }
  ]
}
```

## Exit codes

The exit code is the same with and without `--json`.

| Exit code | Meaning | Error codes |
|---|---|---|
| 0 | Success | |
| 1 | Rejected records or a failed verification | `rejects`, `verification_failed` |
| 2 | Usage error, invalid plan, a state of another target, a state file that cannot be read, or a directory that cannot be written | `usage`, `invalid_plan`, `plan_mismatch`, `target_rejected_schema`, `state_mismatch`, `invalid_state`, `directory_unwritable` |
| 3 | Missing credential or authentication failure | `missing_credential`, `auth_failed` |
| 4 | Source or target unreachable, or a source collection that changed during the read | `source_unreachable`, `target_unreachable`, `source_compacting` |
| 5 | Internal error | `internal_error` |
