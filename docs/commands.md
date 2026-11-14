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
| `--json` | Print one JSON document on stdout, except in the two cases of [JSON output](#json-output). Progress lines and the `Warning:` lines of `copy` and `sync` go to stderr; plan warnings are in the document under `warnings`. |

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
exit 2, `invalid_plan`. A warning does not stop `copy` or `sync`.

`check` finds, among others:

- Names that Weaviate refuses (`invalid_name`). A tenant name has 1 to 64
  letters, digits, `_` or `-`.
- A vectorizer module that the server has not enabled (`module_not_enabled`).
- A missing provider key for a vectorizer: exit 3, `missing_credential`.
- An existing collection whose settings differ from the plan
  (`plan_mismatch`).
- Plan entries that share a collection, one tenant each, with different
  vectors, a different `replication_factor`
  (`conflicting_replication_factor`), different settings for a shared
  property, or property names that differ only in case. See
  [Edits you can make](plan-file.md#edits-you-can-make).
- A `replication_factor` above the node count of the cluster, for a
  collection that does not exist yet (`too_few_nodes`). See
  [replication](weaviate.md#replication).
- Warnings: `missing_property` (an existing collection lacks a planned
  property; `copy` adds it), `coerced_keys` (a metadata key with mixed types
  whose values `copy` converts), `large_ints` (an `int` or `number` property
  for a key that holds ints above 2**53, which `copy`
  [rejects](plan-file.md#types-and-conversion)), `quantizer_training` (a
  trained quantizer whose training limit is above the record count of the
  source), `not_replicated` (with `replication_factor: auto`, an existing
  collection has factor 1 on a cluster of more than one node), and
  `unknown_node_count` (the key may not read the nodes, so `check` did not
  compare a planned factor with the node count, or `auto` has no node count
  for a collection that does not exist yet).

## `copy`

`copy` creates the target collections, then copies the records page by page.
It prints the [replication factor](weaviate.md#replication) of each
collection it creates and lists them under `created_collections` in its JSON
document. With `replication_factor: auto`, it warns when the server chose the
factor and the key may not read the nodes (`unknown_node_count`), or the
factor is below the one that `auto` sent (`low_replication_factor`).

| Option | Effect |
|---|---|
| `--collection NAME` | Copy only the plan entries whose source or target name is NAME. A target name selects every entry that writes to that collection. Repeatable. |
| `--parallelism N` | Reads per collection at the same time, 1 to 8. Default: 5. |
| `--consistency ONE\|QUORUM\|ALL` | The write consistency level. |
| `--yes` | Go on without asking when the target holds objects that `sync --prune` would delete. See [Existing collections](#existing-collections-and-the-recorded-target). |

### Resuming

`copy` records its progress in `state.sqlite`. A read position advances only
after Weaviate has stored the page. When a run stops for any reason, the next
`copy` in the same directory starts at the page that was not stored. Each
object's UUID comes from its source id, so a page written twice replaces the
same objects and adds none.

One `copy` or `sync` at a time can use a directory. While it runs it holds a
lock on `copy.lock`, and a second `copy` or `sync` exits with `usage`. The
operating system releases the lock when the process ends. On a system without
`fcntl`, such as Windows, `copy` and `sync` run without the lock.

### Records added or deleted between runs

Each run also copies the records added to the source since the last run. It
finds its place from the last two records the previous run read. A run that
continues a stopped copy does the same for each part of the collection that
is not finished, before it reads that part's next page:

- When deletions in the source moved those records, the run finds them at
  their new position and reads what follows. After a stopped copy, a record
  that the deletions moved from one part of the collection into another can
  be written twice, unchanged, and is then counted twice as copied.
- When one of them is gone, the run reads again from a lower position: after
  the last two records of an earlier finished part, when it finds them. When
  it finds none, an unfinished part reads again from the end of the nearest
  unfinished part before it. When there is no such part, the run reads from
  the start of the collection and warns, once per collection, "the source
  changed since the last run; re-reading from the start".
  A part that copies records added after the first run reads from the start
  of the collection when one of its own last two records is gone.
  Records read again are written again unchanged.
  The JSON document of such a run gives the lowest position it read again
  from as `reread_from`.
- When the two records were deleted and added back with the same ids, the run
  cannot tell them apart from the originals. It can then skip records that
  no run copied: the records added in between, and after a stopped copy also
  records that were not read yet. It gives no warning. The `ids` check of the
  full `verify` names the skipped records, and `sync` writes them. The
  `count` check alone can miss them when Weaviate still holds objects of
  deleted records. Run the full `verify` after the last copy of a source
  that changed.

### Rejects

`copy` and `sync` append each record that they cannot convert, or that
Weaviate refuses, to `rejects.jsonl` with the reason. They then exit 1,
`rejects`. A reject stays until its cause is fixed: edit the plan or the
record in the source, then run [`sync`](#sync). It reads the record again
and writes it. Objects already in Weaviate are written only when they
changed. `rejects.jsonl` keeps the lines of earlier runs, so a reject that
stays has one line for each run that met it. `status` counts the rejects of
the last `sync` and of the `copy` runs after it.

### Retries

`copy` retries a transient error: a network error, a timeout, HTTP 429,
500, 502, 503 or 504, and the matching gRPC statuses. It tries up to 7 times
and waits 15.75 s to 31.5 s in total between the attempts. A `Retry-After`
header from the source raises each wait to its value, up to 30 s. The
Weaviate client also retries a gRPC `UNAVAILABLE` answer inside each attempt,
for 15 s on a write and 63 s on a read. While Weaviate answers `UNAVAILABLE`,
`copy` can therefore wait about 2 minutes on a write and about 8 minutes on a
read before it stops. It also retries:

- a batch of more than one object in which Weaviate refused every object. A
  batch of one refused object is a reject, unless the error is about a node
  outage or a busy model provider. When a model provider answered an object
  of the batch with another status than 401, 403, 429 or 5xx, `copy` does
  not retry the batch. It sends the objects one by one;
- an object that Weaviate refused with an error about a node, a shard or a
  deadline, as it does while a node restarts;
- an object that Weaviate could not embed because the model provider
  answered HTTP 429 or 5xx, or because the provider's rate limit resets too
  late for the batch. Weaviate does not repeat that request itself. The
  objects of the batch that Weaviate stored are not sent again. Another
  answer of the provider, such as HTTP 400, makes the object a reject at once.

When the retries run out, `copy` stops with `source_unreachable` or
`target_unreachable`, exit 4, and stores no reject. The next run starts at
that page. A provider that still answers 429 or 5xx after the retries also
ends the run with `target_unreachable`, and the message holds the provider's
status: wait for the rate limit to reset, or lower `--parallelism`, then run
`copy` again. `copy` has no option that skips a page or a record. While the
provider answers a record of a page with 5xx, every run stops at that page.
The message holds the provider's status and text. It does not hold the id of
the record. Read the provider's text for the cause and solve it at the
provider. To copy the other collections in the meantime, run `copy` with
`--collection` for each of them. A refused API key is not retried: `copy`
stops with `auth_failed`, exit 3. Another request that the source refuses,
with HTTP 501 or a 4xx status other than 408 and 429, is not retried either:
`copy` stops with `source_rejected`, exit 2, because the same request is
refused again until an input changes. HTTP 408 is not retried either: `copy`
stops with `source_unreachable`, exit 4.

### Existing collections and the recorded target

When a target collection exists, `copy` compares it with the plan. It adds a
planned property that the collection lacks and lists it under
`added_properties` in its JSON document. Any other difference stops it with
`plan_mismatch`, exit 2.

On the first run of a directory into a collection or tenant that exists,
`copy` and `sync` list the ids of the source and of the target once. They
skip a tenant that is not active: Weaviate reads none of its objects, and
the first write stops with the tenant's error. They also skip a collection
without the `source_id` property, which no object can hold yet. When
the target holds objects with a `source_id` that the source does not hold,
they print `Warning: <source>: N objects have a source_id that <source> does
not hold; sync --prune deletes them. ...`. Such objects come from another
source, often another migration directory, or from records deleted from this
source. See [Stale objects and `--prune`](#stale-objects-and---prune).

Before they create or write anything, `copy` and `sync` then ask
`Go on with the copy? [y/N]` in a terminal. Without a terminal, for example
when an agent or a script runs them, they do not ask. Unless the answer is
`y`, or `--yes` was given, they stop with `foreign_objects`, exit 2, and
write nothing. Copy into another collection or tenant instead, or run the
same command with `--yes` when these objects may be deleted. A run that goes
on lists the warning `foreign_objects` in its JSON document.

`state.sqlite` records the target of each source collection: the Weaviate
host and port, the collection and the tenant. Once records of a collection
are copied, `copy` and `sync` stop with `state_mismatch`, exit 2, when
`WEAVIATE_URL` or the plan names another target, or when the target
collection is gone. To copy to another target, use another `--dir`.

`copy`, `sync` and `verify` also stop with `invalid_plan` before they connect when
the source settings name another source than the one in `plan.yaml`. Each
source guide names the settings it compares, for example
[Chroma](sources/chroma.md#connection-settings).

## `sync`

`sync` reads the source again and writes the records that are new or that
changed since the last `copy` or `sync`. With `--prune`, it also deletes the
objects whose source record is gone. It takes the options of `copy` and two
of its own:

| Option | Effect |
|---|---|
| `--collection NAME` | As for `copy`. |
| `--parallelism N` | As for `copy`. |
| `--consistency ONE\|QUORUM\|ALL` | As for `copy`. The deletions of `--prune` use it too. |
| `--yes` | As for `copy`. |
| `--prune` | Delete the objects whose source record is gone. |
| `--force` | With `--prune`, prune a collection with more than 10% stale objects. Without `--prune` it is a usage error. |

### The cutover

`sync` copies the changes your application made to the source since `copy`:

1. `copy`.
2. `sync`, as often as you like, while the application still writes to the
   source.
3. Stop the writes to the source.
4. `sync --prune`, then `verify`.
5. Switch the application to Weaviate.

Some sources need the writes to them paused while `copy` and `sync` run; the
"Writes during a copy" section of each [source guide](sources/README.md)
says which. The writes made between two runs are copied by the next one.

Do not run `sync` against that target after the switch. The source no longer
holds the current data. `sync` would write each record whose object your
application deleted in Weaviate again, and `--prune` would delete the objects
with a `source_id` whose record is not in the source.

### What a sync writes

`copy` stores a hash of each record it writes in `state.sqlite`, made from
the record and the plan entry. `sync` reads every record of the source,
hashes it, and writes the records whose hash differs from the stored one or
has none: new records, changed records and the rejects of earlier runs. The
other records count as `unchanged` and are not written.

Then `sync` lists the ids of the source and of the target. A record without
an object in Weaviate, for example one whose object something other than
this tool deleted, is read from the source by id and written in the same
run. `sync` reads the whole source and lists both sides, so it takes about
as long as a `copy` that writes nothing plus a full `verify` of the ids.
Like `copy`, it holds `copy.lock`; `verify` and `status` run beside it, and
a `verify` during a `sync --prune` can count objects that are being written
or deleted.

A plan entry edited since the last run, for example a renamed property or a
dropped key, describes other objects than the ones in Weaviate, and no
stored hash matches. `sync` then writes every record of that entry again,
prints `Warning: <source>: the plan entry changed since the last copy; every
record is written again`, writes the event `plan_changed` and reports
`plan_changed` for the entry. A `copy` after the edit leaves the warning to
the next `sync`. Edits to a text that does not reach the objects, such as
the reason of a dropped key, change no hash.

### Stale objects and `--prune`

A stale object is an object of the target collection, or of the entry's
tenant, that has a `source_id` and no source record with that id. Every
`sync` counts them and reports them as `stale`. Without `--prune` it deletes
nothing and ends with `Run to-weaviate sync --prune to delete the N stale
objects.` An object without a `source_id`, written by something other than
this tool, is neither counted nor deleted.

`--prune` treats every object with a `source_id` as one that this migration
directory wrote. Give each target collection, or each tenant, one migration
directory and one source: the objects that another directory wrote there
are stale to this one, and `--prune` deletes them. The first `copy` or
`sync` of a directory into an existing collection asks before it goes on
when it finds such objects.

With `--prune`, after it wrote the new and changed records, `sync` reads the
stale records from the source by id. A record that the source holds was
missed by the id listing, for example because the application deleted an
earlier record while the ids were listed; its object stays. For the others,
`sync` forgets the hashes and deletes the objects, 100 per request. A record
that comes back is written again, also after a deletion that failed. When
Weaviate reports objects of a request that it did not delete, `sync` stops
with `delete_failed`, exit 4; run the same `sync --prune` again to delete
the rest. Two checks limit what `--prune` deletes:

- When more than 10% of the objects with a `source_id` in a collection are
  stale, `sync` deletes nothing there, reports `prune_refused` for it and
  ends its line with `prune refused`. Once every collection is done it exits
  1, `prune_limit`, and names the collections. When records were also
  rejected, it exits with `rejects` and that message names them. Check that
  the records were deleted on purpose and that no other migration directory
  or application writes objects with a `source_id` to the collection. To
  delete the objects anyway, run `sync --prune --force`.
- `sync --prune` reads the source count before and after it lists the ids.
  When either differs from the listed count, the source is still written to.
  `sync` stops with `source_writing`, exit 4, and deletes nothing in that
  collection. The collections synced before it keep their writes and
  deletions; their `sync_finished` events are in `events.jsonl`. Stop the
  writes and run the same command again.

The output names, per entry and in total, the records `copied` (written),
`unchanged`, the `rejects`, the `stale` objects and the objects `deleted`.
The JSON document is under [JSON output](#json-output).

## `verify`

`verify` compares each target collection with its source and writes
`report.md` and `report.json`. It runs three checks:

| Check | What it compares |
|---|---|
| `count` | The record count of the source and the object count of Weaviate. |
| `ids` | The full list of ids on both sides. It counts the missing and the extra ids and names up to 20 of each. |
| `records` | A random sample of records: every property, and each kept source vector, rounded to float32, within 1e-5. |

A sampled record that the plan rejects is not compared. When the target
holds no object for it, `records` counts it under `rejects skipped`. When
the target holds one, the record counts as differing, and the detail gives
the reason of the reject. The object was written under another plan or
before the record changed. Fix the cause of the reject and run `sync`.

An existing collection keeps its property types, so a reject that needs
another property type is not fixed by an edit of `plan.yaml`: `sync` stops
with `plan_mismatch`. Copy into a new collection instead. Use another
`--dir`, and give the target a name that does not exist, or delete the
collection first.

A failed check exits 1, `verification_failed`. Its detail names the cause,
and the fix where there is one. Extra objects with a `source_id` lost their
source record, and `sync --prune` deletes them. Extra objects without a
`source_id` were written by something other than this tool; `sync --prune`
leaves them, so delete them in Weaviate if they do not belong there. The
`count` check cannot tell the two apart and points to the `ids` check.

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
rejects, whether the copy is done, and the recorded target. It reads
`plan.yaml` and `state.sqlite` and needs no credentials, so you can run it
during a copy.

The total is the highest of the inspected count and the counts that `copy`
read. The copied count includes records that were deleted from the source
after `copy` wrote them, so after deletions it can be above the total. A
`sync` stores the count it read as the total and counts every record it
read, so after a `sync` the copied count is the total minus the rejects.

## Migration directory

All files of a migration live in one directory, `./.to-weaviate` unless you
pass `--dir`:

| File | Written by | Holds |
|---|---|---|
| `inspect.json` | `inspect` | What the source holds |
| `plan.yaml` | `inspect`, then you | The migration plan |
| `state.sqlite` | `copy`, `sync` | Progress, read positions, the recorded targets, a hash of each written record and of each plan entry |
| `copy.lock` | `copy`, `sync` | The lock of the running `copy` or `sync` |
| `events.jsonl` | `copy`, `sync` | The events of `copy` and `sync`: `collection_started`, `progress` (at most one every 5 s per collection), `compaction_wait`, `source_changed`, `collection_finished`, `plan_changed` and `sync_finished` |
| `rejects.jsonl` | `copy`, `sync` | One line per rejected record and run that met it, with the reason |
| `report.md`, `report.json` | `verify` | The verification report |
| `verify-…/`, `sync-…/`, `ids-…/` | `verify`, `sync`, `copy` | The source and target ids while the command lists them; removed when it ends |

When the disk is full, or the directory or a file in it is read-only, a
command stops with `directory_unwritable`, exit 2. The message names the
cause, and the file when the operating system reports it. `copy` resumes at its page once the directory can be written.

## JSON output

With `--json`, stdout holds exactly one JSON document and every document has
`status`: `success` or `error`. Two cases print no document: an option value
that the parser refuses exits 2 with its message on stderr, and Ctrl-C exits
130 with no output. An error document also has `code`, `message` and `hint`:

```json
{
  "status": "error",
  "code": "auth_failed",
  "message": "Weaviate refused the request: the API key is missing or wrong.",
  "hint": "Check WEAVIATE_API_KEY in the environment or in .env."
}
```

The document of a successful `inspect chroma` has `plan`, `collections`,
`total`, `decisions` and `notes`. `notes` is a list of text lines about the
inspection as a whole. It is empty unless the Chroma server cannot list its
databases. Then it holds one line that names the databases `inspect` read;
see [Supported Chroma versions](sources/chroma.md#supported-chroma-versions).
The same line goes to stderr.

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
  "created_collections": [
    {
      "collection": "Docs_cosine",
      "replication_factor": 3,
      "chosen_by": "nodes",
      "nodes": 3
    }
  ],
  "warnings": [
    {
      "code": "coerced_keys",
      "collection": "Docs_cosine",
      "message": "key code has the types int: 650, text: 50; values are stored as text"
    }
  ]
}
```

A successful `sync --prune`:

```json
{
  "status": "success",
  "prune": true,
  "collections": [
    {
      "ref": "451fdd2c-b3ba-4224-8117-97d6a271e002",
      "source": "docs-cosine",
      "target": "Docs_cosine",
      "tenant": null,
      "copied": 10,
      "unchanged": 690,
      "rejects": 0,
      "stale": 5,
      "deleted": 5,
      "plan_changed": false,
      "prune_refused": false
    }
  ],
  "added_properties": [],
  "created_collections": [],
  "warnings": []
}
```

`created_collections` lists the collections that the run created. `chosen_by`
is `plan` for a factor of the plan, `nodes` for the factor that `auto` took
from the node count `nodes`, and `server` when the request sent no factor.
`server` has two cases. With `nodes` `null`, the key may not read the nodes.
With a number in `nodes`, the server refused the factor that `auto` took from
that node count. `nodes` is also `null` for `plan`: the run does not read the
node count for a factor of the plan.

`copied` counts the records written; `unchanged` the records whose stored
hash matched. The `prune_limit` and `rejects` error documents hold the same
`collections` list.

## Exit codes

The exit code is the same with and without `--json`.

| Exit code | Meaning | Error codes |
|---|---|---|
| 0 | Success | |
| 1 | Rejected records, a failed verification, or a collection with more than 10% stale objects that `sync --prune` left alone | `rejects`, `verification_failed`, `prune_limit` |
| 2 | Usage error, invalid plan, a state of another target, a state file that cannot be read, a directory that cannot be written, a request that the source refused and refuses again until an input changes, or a first run into a collection with objects that `sync --prune` would delete, not confirmed | `usage`, `invalid_plan`, `plan_mismatch`, `target_rejected_schema`, `state_mismatch`, `invalid_state`, `directory_unwritable`, `source_rejected`, `foreign_objects` |
| 3 | Missing credential or authentication failure | `missing_credential`, `auth_failed` |
| 4 | Source or target unreachable, a source collection that the source compacted during the read or did not finish compacting, a source still written to during `sync --prune`, or a deletion that Weaviate did only partly | `source_unreachable`, `target_unreachable`, `source_compacting`, `source_writing`, `delete_failed` |
| 5 | Internal error | `internal_error` |
| 130 | Stopped with Ctrl-C; no output | |
