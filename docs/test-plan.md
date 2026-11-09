# Test plan for to-weaviate

Written against commit fbc0271 (after batches B1 to B8). Servers of the
acceptance suite: Chroma 1.5.9, Weaviate 1.40.4 with `text2vec-transformers`
(384 dimensions), RBAC on with the single key `admin-key`, no
`ASYNC_INDEXING`. The unit suite has 1170 tests, the acceptance suite 52
(about 2 minutes locally, an estimated 10 to 11 minutes in CI), the cloud
suite 1 (release workflow only).

## 1. Scope and quality goals

`to-weaviate` copies Chroma collections into Weaviate: `inspect chroma`
writes `inspect.json` and a draft `plan.yaml`; `check` compares the plan
with the server; `copy` creates the collections and writes the records;
`verify` compares the two sides; `status` reads `state.sqlite`. The tool
never writes to Chroma. This plan covers those five commands, their
options, the plan file, the migration directory, and the two clients.

The goals, in the tool's own terms:

1. No lost record. Every source record is either an object in the target or
   a line in `rejects.jsonl` with a reason. `verify` finds a record that is
   neither.
2. No duplicate. An object is written under one UUID derived from the
   source id; a page written twice replaces, it does not add.
3. No secret in a file or on the terminal. API keys, URL passwords and the
   values of a Chroma embedding function stay out of `plan.yaml`,
   `inspect.json`, `state.sqlite`, the JSON documents, stderr and tracebacks.
4. A copy that resumes. A run that stops at any point, for any reason,
   continues at a page whose earlier pages are all written, in the same
   directory, against the same target.
5. Exit codes a script can trust. 0 success, 1 rejects or a failed
   verification, 2 usage or plan, 3 credential, 4 unreachable or a source
   collection compacted during the read, 5 internal, the same with and
   without `--json`, and one JSON document on stdout.
6. A plan a person can edit. Every allowed edit is applied by `copy`;
   every disallowed edit is refused by `check` and `copy` before anything
   is written, with a message that names the field.

Out of scope: performance targets, the `sync`, `--restart` and prune
commands of later milestones, and Weaviate versions other than 1.40.4.

## 2. Coverage matrix

Columns: U the unit tests (`tests/unit`, fakes), A the acceptance tests
(`tests/acceptance`, both acceptance levels), C the Chroma Cloud suite
(`tests/cloud`). That suite is deleted, because no Chroma Cloud source
exists; column C shows what it covered. Cells: `cov` covered, `part`
partly, `no` not covered, with the tests that cover it. Prefixes: `cmd` =
`test_cli_commands.py`, `json` = `test_cli_json.py`, `acc/x` =
`tests/acceptance/chroma/test_x.py`. A row marked `deferred Fn` is judged in
`triage.md` or the ledger and gets no new case here.

The Level column names the level that proves the row. `unit`: fakes prove
it. `acceptance API`: it needs the real servers and reaches them only
through the command and the server APIs, so it also runs with
`ACCEPTANCE_TARGET=cloud`. `acceptance infra`: it restarts or deletes a
server through docker or kubectl (marker `infra`, local servers only).
`local cloud run`: it needs a Weaviate Cloud cluster, which the API-level
tests reach with `ACCEPTANCE_TARGET=cloud` from a laptop. A row with no
acceptance case, such as a deferred one, is `unit`. The local cloud run
needs a serverless cluster. A sandbox cluster refuses `hnsw`, `flat`,
`dynamic` and the quantizer settings, so those cases fail there with the
server's message.

| Feature or behavior | U | A | C | Level |
|---|---|---|---|---|
| `inspect --database`, `CHROMA_DATABASE`, unknown database | cov `cmd::test_inspect_databases`, `::test_inspect_that_finds_no_collection_*` | no (case A-18) | part `test_cloud` (database from the env) | acceptance API |
| `inspect --collection`, unknown name | cov `cmd::test_inspect_unknown_collection_hint_names_both_database_settings` | part `acc/migration` (known names only) | cov | acceptance API |
| `inspect --sample-pages` | cov `test_chroma_source::test_inspect_sample_pages_limits_metadata_pages` | cov `acc/rejects` | no | acceptance API |
| `--vectors keep/embed/both`, `--vectorizer`, `--vectorizer-option` | cov `test_plan::test_draft_*`, `cmd::test_inspect_vectorizer_option*` | cov `acc/vector_modes`, `acc/resume` | no | acceptance API |
| `--index`, `--quantizer`, `--quantizer-bits`, refused combinations | cov `test_plan` (B8), `cmd::test_inspect_refuses_*` | cov `acc/vector_settings` (24 cases) | no | acceptance API |
| `--force`, existing plan | cov `cmd::test_inspect_refuses_to_overwrite_plan` | no (A-18) | no | acceptance API |
| `check`: plan findings, modules, differences, provider keys | cov `test_check`, `cmd::test_check_*` | part `acc/multi_tenant`, `acc/vector_settings`; module_not_enabled and missing provider key not run (A-09) | cov | acceptance API |
| `copy --collection`, unknown name | cov `cmd::test_copy_collection_matches_source_or_target_name` | no (A-18) | no | acceptance API |
| `copy --parallelism` bounds and effect | cov `cmd::test_parallelism_*`, `test_copy::test_parallelism_is_bounded` | no (default 5 only; A-15) | no | acceptance API |
| `copy --consistency` | cov `cmd::test_copy_creates_the_collection_and_passes_the_consistency` (fake records it) | no (A-15) | no | acceptance API |
| `verify --sample`, `--quick` | cov `test_verify::test_sample_*`, `::test_quick_skips_the_id_comparison` | part `acc/differences` (`--quick`); bounds not run (A-16) | no | acceptance API |
| `status` with and without state, stored target | cov `cmd::test_status_*` | cov `acc/migration`, `acc/multi_tenant` | no | acceptance API |
| `--dir`, `--json` on every command, `--version`, Click errors | cov `json::test_every_command_has_*`, `::test_click_errors_and_version_keep_their_behavior` | part (`--json` always; text mode never, A-19) | no | acceptance API |
| Text mode output and `Error:`/`Hint:` lines | cov `json::test_text_error_goes_to_stderr`, `::test_unexpected_error_in_text_mode` | no (A-19) | no | acceptance API |
| Plan: `version`, unknown field, empty collections, top-level shape | cov `test_plan::test_load_refuses_a_typo`, `cmd::test_plan_without_a_collection_is_refused_on_load` | no | no | unit |
| Plan: duplicate YAML key, merge key, non-UTF-8, YAML error text | cov `test_plan::test_load_*`, `cmd::test_a_yaml_error_*`, `::test_a_plan_that_is_not_utf8_is_an_invalid_plan` | no | no | unit |
| Plan: `target.name` case collisions, names the server refuses | part `cmd::test_check_refuses_target_names_that_differ_only_in_case`; a name outside Weaviate's regex is not checked (A-14) | no | no | acceptance API |
| Plan: `target.tenant`, `tenant: ""` | cov `acc/multi_tenant`, `cmd::test_check_finds_entries_that_write_to_one_place`; `""` deferred F12 | cov | no | acceptance API |
| Plan: vectors (name, mode, vectorizer, distance, index, quantizer fields) | cov `test_plan` (40 tests), `test_schema::test_kwargs_*` | cov `acc/vector_settings`, `acc/vector_modes` | part (default only) | acceptance API |
| Plan: properties (`source_key`, `name`, `type`, tokenization, flags, `coerced_from`) | cov `test_writer::test_to_object_*`, `test_schema::test_kwargs_property_flags` | cov `acc/migration::test_metadata_shapes_map_to_property_types` | part | acceptance API |
| Plan: `dropped`, unmapped key | cov `cmd::test_a_key_that_the_plan_neither_maps_nor_drops_is_an_invalid_plan`, `::test_copy_resumes_after_an_unmapped_key_is_listed_under_dropped` | no | no | unit |
| Plan: allowed hand edits applied end to end (rename, retype, drop, tenant) | part (`write_plan` edits in acceptance only) | part `acc/multi_tenant`, `acc/vector_settings` | no | acceptance API |
| Plan: property added after the collection exists | cov `cmd::test_copy_adds_a_planned_property_that_the_collection_lacks` (U-15) | no | no | unit |
| Source: distances l2, cosine, ip; legacy 0.5.x metadata; spann | cov `test_chroma_config` | cov `acc/vector_settings::test_chroma_distances_map_to_weaviate_distances` | cov (cosine) | acceptance API |
| Source: embedding functions (10 mappings, default, azure, unknown, stored form) | cov `test_chroma_mapping` (29 tests) | part `acc/embedding_secrets` (azure openai) | no | acceptance API |
| Source: metadata types scalar, array, mixed, partial, sparse | cov `test_types`, `test_chroma_source::test_inspect_*` | cov `acc/migration` | part | acceptance API |
| Source: empty collection | cov `test_chroma_source::test_inspect_empty_collection`, `test_verify::test_records_pass_on_an_empty_source` | no (A-03) | no | acceptance API |
| Source: records without embeddings; null inside an embedding | cov `test_chroma_source::test_read_decodes_a_null_in_an_embedding_as_nan`, `cmd::test_copy_rejects_a_record_whose_embedding_holds_a_null` | no (A-03) | no | unit |
| Source: empty strings, unicode, long documents, int64 and float edges, booleans as strings | no (A-10, U-09) | no | no | acceptance API |
| Source: ids that are UUIDs, other strings, colliding after derivation | part `test_ids::test_to_uuid`, `acc/migration::test_end_to_end`; collision not covered (U-07) | part | no | acceptance API |
| Source: vectors of mixed dimension in one collection | no (U-05; Chroma refuses it, section 5) | no | no | unit |
| Source: collection names that need renaming | part `test_naming`, `test_plan::test_draft_unique_collection_names`; non-ASCII untested (U-09, A-11) | no | no | acceptance API |
| Source: several databases, listing order, paging of the collection list | cov `cmd::test_inspect_of_every_database_*`, `test_chroma_client::test_collections_*` | part (one database) | no | acceptance API |
| Source: a page shorter than the limit | cov `test_cli_chroma::test_a_page_shorter_than_the_limit_does_not_end_a_partition` (U-01) | no | no | unit |
| Source: records deleted during a copy | no; deferred F4 (U-08 pins the documented limit) | no | no | unit |
| Source: records added between runs (tail) | cov `test_copy::test_*tail*`, `cmd::test_every_copy_reads_the_records_added_since_the_last_run` | cov `acc/migration::test_each_copy_adds_the_records_that_the_source_gained` | no | acceptance API |
| Source: compaction on distributed Chroma (wait for num_unindexed_ops 0, stop on a changed version or log_position, single-node HTTP 500) | cov `test_chroma_client::test_indexing_status_*`, `::test_another_500_on_indexing_status_is_retried`, `::test_collection_by_name_reads_the_collection_document`, `test_chroma_source::test_wait_compacted_*`, `::test_compaction_state_of_a_collection_created_again_names_inspect_force`, `test_copy::test_the_wait_comes_before_the_first_page_and_the_state_after_the_last_read`, `::test_a_changed_state_stops_the_copy_and_the_next_run_reads_from_the_start`, `::test_a_source_without_compaction_is_not_compared`, `::test_the_wait_writes_a_compaction_wait_event_per_change`, `::test_a_wait_that_ends_without_compaction_reads_no_page`, `test_verify::test_verify_waits_before_the_counts_and_reads_the_state_after_the_records`, `::test_verify_stops_with_source_compacting_and_writes_no_report`, `test_cli_chroma::test_copy_and_verify_wait_until_chroma_compacted`, `::test_a_single_node_chroma_gets_one_indexing_status_request_per_command` | cov `acc/source_order::test_copy_during_compaction` (cluster leg) | no | acceptance API |
| Source: `indexing_status` answers HTTP 404, from Chroma before 1.4.1 or for a missing collection: no guard, one request, no retry (ruling R103) | cov `test_chroma_client::test_a_404_on_indexing_status_is_none_and_not_retried` | no (the compose Chroma has the route) | no | unit |
| Source: `copy` and `verify` against a Chroma without the `indexing_status` route copy every record with one request per command (ruling R103) | cov `test_cli_chroma::test_a_chroma_without_the_indexing_status_route_copies_and_verifies_unguarded` | no (the compose Chroma has the route) | no | unit |
| Source: a not-implemented answer after a successful first poll keeps the guard off (ruling R101) | cov `test_chroma_source::test_a_later_not_implemented_answer_keeps_the_guard_off` | no | no | unit |
| Source: 5xx, 429 with Retry-After, transport error, redirect, non-JSON | cov `test_chroma_client` (client level only; U-02 adds the copy path) | no (A-04) | no | unit |
| Source: 401/403, missing key, Chroma Cloud host rule, token header forms | cov `json::test_source_errors_map_to_codes`, `test_settings`, `test_chroma_client::test_sends_token_header` | part (no key on the local server) | cov | unit |
| Source: tenant from `/auth/identity` (`_chroma_source`) | no (U-13) | cov (implicit) | cov | acceptance API |
| Target: new collection, tenants, properties union | cov `test_schema::test_ensure_collection_*` | cov | cov | acceptance API |
| Target: existing collection with differences, with extra items, with multi-tenancy | cov `test_schema::test_differences_*` (16 tests), `json::test_copy_stops_when_the_collection_exists_with_other_settings` | no (A-12) | no | acceptance API |
| Target: existing name that differs only in case | no; deferred (ledger Task 10 minor) | no | no | unit |
| Target: index types, quantizers, trained quantizers, dynamic | cov `test_schema`, `cmd::test_copy_enables_*` | cov `acc/vector_settings`, `acc/multi_tenant` (dynamic xfails only when the server refuses the dynamic index) | no | acceptance API |
| Target: module not enabled | cov `cmd::test_check_refuses_module_not_enabled_on_target` | no (A-09) | no | acceptance API |
| Target: 401, 403 at connect | cov `json::test_target_errors_map_to_codes` | cov `acc/migration::test_wrong_or_missing_api_key_exits_3` | no | acceptance API |
| Target: 403 at create or at insert (RBAC without write rights) | no (U-04) | no (needs a second key, section 5) | no | unit |
| Target: writes refused (READONLY shard, inactive tenant, every object refused) | cov `cmd::test_copy_stops_when_weaviate_refuses_every_object_of_a_batch` (fake) | no (A-02, A-06) | no | acceptance API |
| Target: per-object refusal with the server's message | cov `test_writer::test_writer_inserts_and_maps_errors` (writer level; U-05 adds the copy path) | no (not producible from Chroma) | no | unit |
| Target: 429/503/5xx, gRPC UNAVAILABLE, WeaviateRetryError, timeouts | cov `test_schema::test_transient_*`, `test_writer::test_writer_retries_*`, `json::test_copy_retries_a_retry_error_and_then_reports_target_unreachable` | no (A-04 covers unreachable only) | no | acceptance infra |
| Target: one object above the gRPC message limit | cov `cmd::test_an_object_above_the_message_limit_is_a_reject` (U-06) | no (A-05 not implemented: no Chroma record reaches the limit) | no | unit |
| Target: cloud connect (port 443, no gRPC URL), explicit port refused | cov `test_schema::test_connect_cloud*`, `test_settings::test_weaviate_url_port_other_than_443_needs_a_grpc_url` | no | cov | local cloud run |
| Error codes 0, 1, 2, 3 | cov `json::test_error_exit_covers_every_code` and per-code tests | cov | cov (0) | acceptance API |
| Error codes 4, 5 | cov `json::test_source_errors_map_to_codes`, `::test_a_compaction_during_copy_is_source_compacting_exit_4`, `::test_a_collection_that_chroma_does_not_compact_is_source_compacting_exit_4`, `::test_unexpected_error_gives_an_internal_error_document` | no (A-04 for 4; 5 by design not provoked) | no | acceptance API |
| JSON document shape under every error (`status`, `code`, `message`, `hint`, fields) | cov `json::test_json_error_document`, `::test_json_stdout_is_one_document` | part (`e2e.run` checks one document; not the keys, A-17) | part | acceptance API |
| Resume: killed process, lock release | cov `test_lock` | cov `acc/resume` | no | acceptance API |
| Resume: refused batch, refused schema, refused quantizer update | cov `cmd::test_copy_resumes_at_the_refused_page_and_writes_each_object_once`, `::test_a_refused_quantizer_update_stops_copy_and_resumes_without_pages` | part (`acc/vector_settings` resume after check) | no | acceptance API |
| Resume: renamed target, other URL, missing collection or tenant, other source | cov `cmd::test_copy_refuses_*` (8 tests) | cov `acc/target_change` | no | acceptance API |
| Resume: state file of an older schema | cov `cmd::test_copy_resumes_a_state_without_targets_and_stores_the_target`, `test_state::test_state_file_without_*` | no | no | unit |
| Resume: corrupted state file, state of a newer schema | cov `cmd::test_a_corrupted_state_file_is_an_invalid_state`, `::test_a_state_file_of_a_newer_schema_resumes` (U-12) | no | no | unit |
| Concurrency: two `copy` processes | cov `test_lock`, `cmd::test_copy_refuses_a_directory_that_another_copy_holds` | part `acc/target_change` (lock held by the test process; A-01 adds a real second process) | no | acceptance API |
| Concurrency: workers and sqlite, stop after an error | cov `test_state::test_lock_keeps_statement_spans_from_overlapping`, `test_copy::test_other_workers_stop_after_a_write_raises` | part (parallelism 5 in every run) | no | acceptance API |
| Secrets: API keys in stdout, stderr, traceback, error bodies | cov `json::test_no_secret_in_any_output`, `::test_secret_in_an_exception_is_redacted`, `::test_debug_prints_the_traceback_on_stderr` | part (no suite-wide check, A-08) | no | acceptance API |
| Secrets: URL password, token userinfo, query, fragment | cov `json::test_url_*`, `test_settings::test_stored_url*` | no (the suite needs exact URLs) | no | unit |
| Secrets: embedding function config, vectorizer URL options | cov `json::test_embedding_config_secrets_are_not_stored_or_sent`, `test_schema::test_kwargs_send_every_url_option_without_userinfo_query_or_fragment` | cov `acc/embedding_secrets` | no | acceptance API |
| Secrets: provider key sent as a header and never printed | cov `test_settings::test_provider_key_becomes_header` | no (A-09) | no | acceptance API |
| Secrets: `--vectorizer-option` value stored as typed; reject reasons unredacted | no; deferred (B4 minor, F26 rest) | no | no | unit |
| Chroma Cloud: 300-record page cap, rate limit, key scope, `CHROMA_TENANT` | cov `test_chroma_client::test_get_refuses_page_over_300`, `::test_honors_retry_after` | n/a | part `test_cloud` (one collection, one run) | unit |

## 3. Risks

Ranked by what a person loses when the risk is real.

- R1 A record is lost or written twice without a message. Paths: a source
  page shorter than the limit ends a partition (`read`, `ids`), a fault in
  the middle of a page, a refusal that the retry hides, two source ids with
  one UUID, a record deleted during the copy (deferred F4), an object the
  server cannot take that blocks the page for good, two processes on one
  directory. `verify` is the second line; its gaps count here too.
- R2 A secret reaches a file or the terminal: an API key in an error body,
  a provider key in a hint, a URL password in a note, a value of an
  embedding function in `plan.yaml`.
- R3 The target does not hold what the source holds: a value changed by a
  type coercion or by the server (int64, float edges, empty strings and
  lists, unicode), a property or collection name the server refuses or
  that collides, an index or quantizer other than the plan's, a schema
  created from a stale inspection.
- R4 The exit code or the JSON document is wrong for the path: a usage or
  environment problem reported as `internal_error`, a missing key in an
  error document, text on stdout in `--json` mode.
- R5 A person cannot proceed: a plan edit that leaves no path (a property
  added after the collection exists), a hint that names the wrong fix, a
  collection without vectors drafted in keep mode, a state that never
  reads done.

## 4. Test cases to add

Order: by risk, then cheapest first. Ids: `U-` unit, `A-` acceptance,
`L-` local cloud run. Each case names its level: `unit`, `API`
(acceptance API), `infra` (acceptance infra) or `local cloud run`
(`make acceptance-cloud`).
Acceptance estimates are local seconds; CI runs about 1.9 times slower;
the acceptance additions have about 8 CI minutes (about 250 local
seconds). The cases marked "optional" come last; stop before them when
the budget is used.

Shared data (added once, in `tests/acceptance/chroma/seed.py`): the
collections `empty` (0 records), `edge` (about 40 records of 8
dimensions, cosine, see A-10), `Mixed-Case`, `mixed-case`,
`1-leading-digit` and `class` (5 records of 8 dimensions each), and the
optional `bulk` (50,000 records of 32 dimensions). New targets go into
`SUITE_COLLECTIONS`: `Empty`, `Edge`, `Mixed_Case`, `Mixed_case_2`,
`C_1_leading_digit`, `Class`, `Bulk`. A recreated
collection can show stale array keys from Chroma's sqlite (ledger fact), so
no case asserts the exact key set of a source collection.

Shared unit helper (new, `tests/unit/chroma_server.py`): a fake Chroma
server for `httpx.MockTransport` that serves `/api/v2/auth/identity`,
`/tenants/{t}/databases`, `.../collections` (paged by offset), `.../count`,
`.../get` (offset, limit, include, honoring an optional `page_cap`) and
`.../query` from an in-memory record list, with `faults` keyed by request
number or by (path, offset) that return an `httpx.Response` or raise an
`httpx.TransportError`. A source factory for `build_app` wraps
`ChromaSource(ChromaClient(settings, transport=MockTransport(server.handler),
sleep=no_sleep), tenant)` so the CLI tests run the real client and source.

### R1: lost or duplicated records

**U-01 A page shorter than the limit ends a partition early.** Level unit,
risk R1. Preconditions: the fake Chroma server with `page_cap=100`, 1000
records. Steps: `inspect chroma`, `copy --parallelism 1`, then `verify`.
Expected today (suspected bug): `read` treats `len(records) < limit` as the
end of the partition, so `copy` writes 100 records, exits 0 with
`copied 100 of 1000`, and `ids()` stops after its first page, so `verify`
reports `missing 0` and only the count check fails. Expected after a fix:
a bounded partition ends only at its stop and a tail or `ids()` only at an
empty page (the rule R49 gave `collections()`); copy writes 1000 records,
one read more per tail. Also run with `page_cap=None` to pin the number of
reads (4 per partition of 1000 at parallelism 1, no extra read for a
bounded partition). Data: records `id-0000..0999`, 2-dimensional vectors.

*Implemented: `tests/unit/test_cli_chroma.py::test_a_page_shorter_than_the_limit_does_not_end_a_partition`. Found the bug; fixed by ruling R52 (a page with no new row is a ChromaError).*

**U-02 A transport error, a 503, a 429 and a 500 in the middle of a page.**
Level unit, risk R1. Preconditions: the fake Chroma server without a page
cap, 2700 records, `--parallelism 3` (3 partitions of 900, 3 pages of 300
each). Faults, one case each: `httpx.ReadError` on the second `/get` of
partition 2; a 503 then a 429 with `Retry-After: 2` on that request; a 500
on all seven attempts. Expected: the first two cases exit 0, every id is written once
(`FakeWriter.writes_per_uuid` all 1), the sleep list holds the backoff
delays and `2.0` for the 429; the third exits 4 `source_unreachable` with
`Chroma returned HTTP 500` in the message, the other two partitions
finished, partition 2's cursor at its first page; a second `copy` without
the fault exits 0 and writes the remaining ids once. Coverage test; the
client-level retry tests cover the delays, this covers the copy path.

*Implemented: `tests/unit/test_cli_chroma.py::test_a_fault_in_the_middle_of_a_page_is_retried`, `::test_a_page_that_fails_every_attempt_stops_copy_and_resumes`. Coverage. After the 500, the other partitions stop before their next page, so whether they finish depends on timing; the test asserts that each cursor follows the pages written.*

**U-03 Chroma fails on the count after the pages.** Level unit, risk R1.
Preconditions: as U-02; fault: 500 on the `/count` request that follows the
last page. Expected: exit 4, every partition finished, no tail; the next
`copy` sends no `/get` for the finished partitions, reads the count, adds a
tail when the source grew (grow the server by 50 records first) and copies
exactly those 50. Coverage test.

*Implemented: `tests/unit/test_cli_chroma.py::test_a_count_that_fails_after_the_pages_resumes_without_pages`. Coverage.*

**U-04 Weaviate refuses writes with 403 (RBAC without write rights).**
Level unit, risk R1. Two cases. (a) `FakeSchema(ensure_error=
InsufficientPermissionsError(httpx.Response(403)))`: `copy` exits 3
`auth_failed`, message `Weaviate refused the request: the API key lacks a
permission.`, hint names `WEAVIATE_API_KEY`, nothing written, `status`
shows copied 0 and no stored target. (b) `WeaviateWriter` over
`FakeInsertClient(fail=lambda n: InsufficientPermissionsError(...) if n == 2
else None)` with 250 records at parallelism 1: exit 3, 100 objects
written, no retry of the 403 (`calls == 2`); then `client.fail = None`,
`copy` exits 0, 250 objects, every uuid written once. Coverage test;
pins that a 403 is not retried and that the page is not advanced.

*Implemented: `tests/unit/test_cli_commands.py::test_copy_creates_nothing_when_the_key_may_not_create_the_collection`, `::test_a_403_on_insert_is_not_retried_and_copy_resumes_at_its_page`. Coverage.*

**U-05 Weaviate refuses one object of a batch; vectors of mixed dimension.**
Level unit, risk R1. Preconditions: `FakeInsertClient` extended with a
`reject` hook that returns `errors={index: SimpleNamespace(message=...)}`
for the object whose vector has another length; records: 20 with
2-dimensional vectors and one (`id-7`) with 1 dimension. Expected: `copy`
exits 1 `rejects`, `rejects.jsonl` has one line with `source_id: id-7` and
the server's message, the other 20 objects are written, `verify` fails the
count and names `id-7` in the ids check. Coverage test; it also records
that the tool does not check dimensions itself (the server does).

*Implemented: `tests/unit/test_cli_commands.py::test_an_object_that_weaviate_refuses_is_a_reject_with_its_message`. Coverage.*

**U-06 One object above the gRPC message limit blocks the page.** Level unit,
risk R1 and R5. Preconditions: `FakeInsertClient` whose `insert_many` for a
chunk of one object raises what the client raises for `RESOURCE_EXHAUSTED`
(`raise_grpc_failure("RESOURCE_EXHAUSTED", WeaviateBatchError, batch=True)`);
records: 5 small ones and one whose document is 5 MiB (`chunks` puts it
alone). Expected today (suspected bug): the chunk is retried 6 times and
`copy` exits 4 `target_unreachable` with the hint `Fix the cause and run
to-weaviate copy again. It resumes at this page.`, which a rerun cannot
fix; the 5 small records are written. Expected after a fix: a chunk of one
object refused with `RESOURCE_EXHAUSTED` becomes a reject with a reason
that names the size, exit 1. The implementer records which; a change needs
a ruling because `transient` is shared with the reader.

*Implemented: `tests/unit/test_cli_commands.py::test_an_object_above_the_message_limit_is_a_reject`. Found the bug (exit 4 with the Weaviate hint after six attempts); fixed by ruling R53.*

**U-07 Two source ids with one UUID.** Level unit, risk R1. Preconditions:
`monkeypatch` `to_weaviate.target.writer.to_uuid` and
`to_weaviate.engine.verify.to_uuid` so that `a` and `b` map to one UUID;
3 records `a`, `b`, `c`. Expected (pins the current behavior; the ids-check
gap is a deferred Task 11 minor, not a B9 bug): `copy` exits 0 with
`copied 3`, the writer holds 2 objects, `verify` exits 1: count fails
(`source count 3, target count 2`), ids passes (`missing 0, extra 0`),
records fails naming `a` or `b` with `property source_id differs`.

*Implemented: `tests/unit/test_cli_commands.py::test_two_source_ids_with_one_uuid_fail_verify`. Coverage; pins the current behavior.*

**U-08 A record deleted from the source during the copy.** Level unit, risk
R1. Preconditions: `FakeSource` with a hook that removes record `id-150`
after the first page is read; 400 records, `page_size` 100, parallelism 1.
Expected (pins the documented limit, deferred F4): `copy` exits 0 and one
record (`id-200`, shifted into the gap) is never read; `verify` exits 1
with `missing 1` naming it. The case exists so that a later fix has a
failing test.

*Implemented: `tests/unit/test_cli_commands.py::test_a_record_deleted_during_the_copy_shifts_another_one_past_the_cursor`. Coverage; pins the documented limit. The record is deleted after the page that holds it is read; then id-200 is missing and id-150 is extra.*

**A-01 A second `copy` process while the first runs.** Level API, risk R1,
+2 s. Test change to `acc/resume::test_copy_resumes_after_the_process_is_killed`:
once `state.sqlite` exists and before the kill, run `copy` in the same
directory. Expected: exit 2 `usage`, message `Another to-weaviate copy
uses <dir>.`, the first process still running (`poll() is None`) and its
`copied` higher at the next poll.

*Implemented: change to `tests/acceptance/chroma/test_resume.py::test_copy_resumes_after_the_process_is_killed`. Coverage.*

**A-02 The target refuses writes (READONLY shard) and the copy resumes.**
Level API, risk R1, 25 s. Preconditions: tiny.l2 copied (10 records);
`add_to_tiny_l2(10, 5)`; the shard of `Tiny_l2` set to `READONLY` with the
admin client (`collection.config.update_shards(status="READONLY")`).
Steps: `copy`; `status`; set `READY`; `copy`; `verify`; cleanup deletes
the 5 added ids. Expected: the first `copy` exits 4 `target_unreachable`,
message starts `Weaviate refused every object of a batch of 5 in Tiny_l2`
(or, if the server refuses the whole request instead of each object, the
same code from `transient`'s retry exhaustion), hint is the resume hint;
`status` shows copied 10, not done; the second `copy` exits 0, 15 objects;
`verify` exits 0. Run time is the retry budget (at least 15.75 s, at most 31.5 s).

*Implemented: `tests/acceptance/chroma/test_faults.py::test_a_read_only_shard_stops_copy_and_the_next_copy_resumes`. Coverage; Weaviate 1.40.4 refuses each object with "store is read-only due to: manually set by user".*

**A-03 An empty collection and a collection without embeddings.** Level API,
risk R1, 10 s. Two cases. (a) `empty`: `inspect` reports count 0 and
dimension null; `check`, `copy`, `verify` exit 0; `status` shows total 0,
done; the collection `Empty` exists with 0 objects. (b) `no-vectors` in
keep mode: `inspect` exits 0; `copy` exits 1 `rejects` with 12 rejects
`no vector` and 0 copied; `verify` exits 1 with the count check failed and
the records detail ending `every sampled record is rejected at copy time;
read rejects.jsonl`. Then in a new directory with `--vectors embed
--vectorizer text2vec-transformers`: `copy` and `verify` exit 0, 12
objects, `near_vector` with the stored vector returns its record first and `near_text`
with one document returns a record.
Suspected (R5, see U-16): the keep-mode draft gives no note that the
source has no vectors.

*Implemented (a): `tests/acceptance/chroma/test_source_shapes.py::test_an_empty_collection_copies_and_verifies`. Coverage. (b) Not implemented: Chroma 1.5.9 answers 422 to an add without embeddings, so no source holds such records; U-16 covers the tool's side.*

**A-04 Chroma and Weaviate unreachable.** Level API, risk R1 and R4, 15 s.
Steps: `inspect chroma --collection tiny.l2` with `CHROMA_URL=
http://localhost:1`; `check` (after a normal inspect) with
`WEAVIATE_URL=http://localhost:1` and `WEAVIATE_GRPC_URL=http://localhost:2`.
Expected: exit 4 `source_unreachable`, message starts `Chroma could not be
reached: ConnectError`, hint names `CHROMA_URL`, after the seven attempts
(at least 15.75 s, at most 31.5 s in total); exit 4 `target_unreachable`, hint names `WEAVIATE_URL`, within
a few seconds; no file other than the inspection in the directory.

*Implemented: `tests/acceptance/chroma/test_faults.py::test_an_unreachable_source_or_target_exits_4`. Coverage.*

**A-05 One record with a document above the gRPC limit (optional).**
Level API, risk R1, 15 s. Preconditions: a collection `big-doc` (seed) with 3
records, one of them a 12 MiB ASCII document; Weaviate's default gRPC
message limit is above 10 MiB, so adjust the size to the limit the server
reports if the first run passes. Expected today: the twin of U-06, exit 4
after the retries (at least 15.75 s), 2 objects written, a rerun exits 4 again.
Expected after the U-06 fix: exit 1 with one reject. Skip when the Chroma
server refuses the add (body size limit); then U-06 stands alone.

*Not implemented: the gRPC limit is 104858000 bytes (client 4.24.1 refuses a larger message with RESOURCE_EXHAUSTED). Chroma 1.5.9 accepts a record with a 32 MiB document and answers 413 to 40 MiB and to 101 MiB (measured), so no source record reaches the limit. U-06 stands alone.*

**A-06 An inactive tenant refuses writes (optional).** Level API, risk R1,
20 s. Preconditions: the plan of `acc/multi_tenant` (tenant-a, tenant-b
into `Tenants`); `copy --collection tenant-a`; tenant `b` created
`INACTIVE` with the admin client. Steps: `copy --collection tenant-b`;
activate `b`; `copy --collection tenant-b`; `verify`. Expected: exit 4
`target_unreachable` with the batch message; then exit 0, 20 objects in
`b`, `verify` exit 0.

*Implemented: `tests/acceptance/chroma/test_faults.py::test_an_inactive_tenant_stops_copy_and_the_next_copy_resumes`. Coverage.*

**A-07 A 50,000-record collection (optional).** Level API, risk R1, 50 s.
Steps on `bulk`: `inspect --sample-pages 5`, `check`, `copy --parallelism 8`,
`verify` (full), `status`. Expected: all exit 0, 50,000 objects, the ids
check walked 50 batches of 1000, the records check `sampled 300,
compared 300`. Covers more than one page per partition and the scratch
sqlite above `ID_BATCH`; `resume` (8,000) already covers both at a smaller
size, so this comes last.

*Implemented: `tests/acceptance/chroma/test_source_shapes.py::test_a_large_collection_copies_and_verifies`. Coverage. Measured locally: the test takes about 10 s and the seed of `bulk` about 15 s more; `bulk` is not in `COLLECTIONS`, so the test that copies every collection does not copy it.*

### R2: secrets

**A-08 Suite-wide secret guard.** Level API, risk R2, 0 s. Test change to
`tests/e2e.py::run` and `::start` and a conftest fixture: after every
command, `admin-key` and every made-up key a test passes in `env` must not
appear in stdout or stderr; at teardown, in no file under the test's
directory (`plan.yaml`, `inspect.json`, `events.jsonl`, `rejects.jsonl`,
`report.md`, `report.json`, `first-copy.json`, and `state.sqlite` read as
bytes). Expected: passes at once; it turns every later case into a secret
test.

*Implemented: `tests/e2e.py::run`, `::run_text` and the fixture `tests/acceptance/conftest.py::_no_secret_in_a_file`. Coverage.*

**A-09 Provider key required, never printed; module not enabled.** Level API,
risk R2 and R4, 3 s. Steps on tiny.l2: `inspect --vectorizer
text2vec-openai`; `check` without `OPENAI_API_KEY`; `check` with
`OPENAI_API_KEY=made-up-openai-key-b9`; in a second directory `inspect
--vectors embed` (default vectorizer `text2vec-weaviate`) and `check`.
Expected: exit 3 `missing_credential`, message `OPENAI_API_KEY is not set`;
exit 2 `invalid_plan` with one error `module_not_enabled` whose message
lists `text2vec-transformers` among the enabled modules, the made-up key in
no output (A-08); exit 2 `module_not_enabled` for `text2vec-weaviate`.

*Implemented: `tests/acceptance/chroma/test_check.py::test_check_needs_the_provider_key_and_an_enabled_module`. Coverage. Weaviate 1.40.4 enables every API-based module by default, text2vec-openai and text2vec-weaviate among them, so check with the made-up key exits 0. The module case picks the first of `CANDIDATE_MODULES` that `/v1/meta` does not list (`text2vec-ollama` on CI, `text2vec-transformers` on Weaviate Cloud).*

### R3: schema and data fidelity

**U-09 Names from non-ASCII, empty and reserved-looking keys.** Level unit,
risk R3. `property_name` and `collection_name` with: `日本` and `中国`
(expected `__` and `___2`); `""` (`p_`); `1st key` (`p_1st_key`); `my key`
(`my_key`); `ID`, `Id`, `id` in that order (`iD`, `id_meta`, `id_meta_2`);
`_additional` (`_additional_meta`); `__meta` (`__meta`); collection `文档`
(`C___`), `1-leading-digit` (`C_1_leading_digit`), `Mixed-Case` then
`mixed-case` (`Mixed_Case`, `Mixed_case_2`). Suspected: a name that starts
with `__` is a GraphQL-reserved prefix and `iD` differs from the reserved
`id` only in case; whether Weaviate 1.40.4 accepts them is read from A-10.
If it refuses them, `property_name` prefixes `p_` for a `__` start and
treats reserved names ignoring case, and this test changes to the new
names.

*Implemented: `tests/unit/test_naming.py::test_drafted_property_names_are_names_the_server_accepts`, `::test_a_reserved_name_near_the_length_limit_falls_back_to_a_numbered_key`, `::test_drafted_collection_names_in_order`. Weaviate 1.40.4 accepts `__` and `iD`. Found the bug: the client refuses the property name `vector`, and the server a name longer than 231 characters (ruling R57).*

**U-10 Allowed hand edits are applied end to end.** Level unit, risk R3.
One parametrized test over the draft of `cli_records(30)`: rename property
`year` to `publication_year`; `year` from `int` to `text` with tokenization
`field`; `year` from `int` to `int[]`; `searchable: true` on `document`
off and `filterable: true` on; `#uri` moved under `dropped` and its
property removed; `distance: l2-squared` with `index.type: hnsw` and
`options: {ef: 32}`; `notes` edited; the entry reordered after a second
one. Expected for each: `check` exit 0, `copy` exit 0, the stored object
holds the property under the new name, type and shape (`2003` as text
`"2003"`, as `[2003]`), no `uri` property, `verify` exit 0, and
`FakeSchema.ensured` holds the collection once. Disallowed edits are
covered already (`test_plan`, `cmd`); this case adds none.

*Implemented: `tests/unit/test_cli_commands.py::test_an_allowed_edit_of_the_plan_is_applied_end_to_end`. Coverage.*

**U-11 An all-digit `--vectorizer-option` value becomes an int.** Level unit,
risk R3. `inspect --vectorizer text2vec-openai --vectorizer-option
model=007 --vectorizer-option dimensions=0256`. Expected today (suspected):
`plan.yaml` holds `model: 7` and `dimensions: 256`. Expected after a fix:
a value is an int only when it has no leading zero and the option is not
`model`. Small; record it if left.

*Implemented: `tests/unit/test_cli_commands.py::test_an_all_digit_vectorizer_option_keeps_its_text`. Found the bug; ruling R57 keeps each value as text in plan.yaml, and the collection is created with it as the type the client declares.*

**A-10 Values at the type limits, unicode, empty and long values.** Level API,
risk R3, 12 s. The `edge` collection (seed): ids `edge-ключ-1`, `edge-日本-2`,
`edge-🙂-3` and ASCII ones; documents: `""`, a 200 KB ASCII text, CJK with
combining marks and an RTL run, one `null`; metadata keys and values:
`int_edge` in {0, -1, 2**53 + 1, 2**63 - 1, -(2**63)}, `float_edge` in
{1e308, -1e308, -0.0, 5e-324, 0.1}, `flag` with `true`, `false`, `"true"`,
`"false"` across records (mixed: text), `num` with `3` and `2.5` (number),
`empty` with `""`, `tags` with `["a"]` and `["a","b","c"]`, and the keys
`ID`, `Id`, `id`, `_additional`, `document`, `uri`, `vector`, `class`,
`1st key`, `my key`, `日本`, `created_temp`, `a__backup`, `x_nullState`
with short text values. If Chroma accepts `tags: []`, one record has it.
Steps: `inspect --collection edge`; `check`; `copy`; `verify --sample
10000`; read the objects back with the admin client. Expected: `inspect`
drafts the names of U-09 (`iD`, `id_meta`, `id_meta_2`, `_additional_meta`,
`document_meta`, `uri_meta`, `vector`, `class`, `p_1st_key`, `my_key`,
`__`, `created_temp_meta`, `a__backup_meta`, `x_nullState_meta`); `check`
exits 0 with `coerced_keys` warnings for `flag` and `num` and no
`reserved_suffix` warning; `copy` exits 0 with 0 rejects; `verify` exits 0;
read back: `2**63 - 1`, `-(2**63)` and `2**53 + 1` equal, `1e308` and
`5e-324` equal, `-0.0` reads `0.0` or `-0.0`, the `""` document reads `""`,
`empty` reads `""`, the 200 KB document has the same length, unicode ids and
documents are byte-equal, `flag` holds the texts `true`/`false`, `tags: []`
reads `[]`. Suspected: (1) Weaviate may return an empty text or an empty
array as null, which fails the records check with `property X differs`;
(2) an int above 2**53 may read back changed (stored or returned as a
double); (3) the server may refuse `__` or `iD` at create, which exits 2
`target_rejected_schema` with the hfresh hint. For each, the tool's side
is a rename rule in `property_name` or a comparison rule in `verify`; the
implementer records the measured server behavior.

*Implemented: `tests/acceptance/chroma/test_source_shapes.py::test_values_at_the_limits_of_their_type_are_copied`. Found two bugs: the property `vector` stopped check and copy with internal_error, and verify reported an int above 2**53, which the server returns rounded to a double. Measured: Weaviate stores an `int` as a double. 2**53 + 1 reads back as 2**53. The client sends the value in a protobuf Struct, so 2**63 - 1 and 2**63 - 2 arrive as the double 2**63; the server takes it only when Go's int64 conversion gives it back, which is true on arm64 (it saturates, and the value reads back as 2**63 - 1 over gRPC and 9223372036854776000 over REST) and false on amd64 (the CI runner refused the 8 records: "requires an integer, the given value is '9.223372036854776e+18'"). Ruling R60: copy rejects an int whose double is 2**63 or above, so the 8 records of `edge` with 2**63 - 1 are rejects on every server; -(2**63) is stored. `""` and `[]` read back as such, Chroma returns -0.0 as 0.0 and drops `tags: []`. The README says that an int above 2**53 is stored rounded, that such an int of 2**63 is rejected, and that a `text` property keeps the digits (rulings R59, R60).*

**A-11 Collection names that need renaming.** Level API, risk R3, 8 s. Steps:
`inspect --collection Mixed-Case --collection mixed-case --collection
1-leading-digit --collection class`; `check`; `copy`; `verify`. Expected:
targets `{Mixed_Case, Mixed_case_2}` (order follows Chroma's listing),
`C_1_leading_digit`, `Class`; all exit 0; the four collections exist with 5
objects each. A unicode collection name is not in this case: Chroma 1.5.9
refuses it at create (section 5).

*Implemented: `tests/acceptance/chroma/test_source_shapes.py::test_collection_names_that_weaviate_refuses_are_renamed`. Coverage.*

**A-12 The target exists with another schema, or with more.** Level API,
risk R3, 6 s. (a) Create `Tiny_l2` with the admin client with `position`
as `text`; `check`; `copy`. Expected: `check` exits 2 `invalid_plan` with
one `plan_mismatch` error `property position: type is text, plan has int`;
`copy` exits 2 `plan_mismatch`; 0 objects. (b) Delete it; create `Tiny_l2`
from the plan's properties plus a property `note` (text) and the plan's
vector config; `copy`; `verify`. Expected: exit 0, 10 objects, `verify`
exit 0, the property `note` still there.

*Implemented: `tests/acceptance/chroma/test_plan_edits.py::test_a_target_with_another_property_type_stops_check_and_copy`, `::test_a_target_with_one_more_property_takes_the_copy`. Coverage.*

**A-13 A hand-edited plan end to end.** Level API, risk R3, 5 s. Edit the
tiny.l2 draft: `position` renamed `pos`, type `text`, tokenization `field`,
`range_filters: false`; the `uri` property removed and `#uri` listed under
`dropped`. Steps: `check`, `copy`, `verify`. Expected: all exit 0; the
collection has `pos` as text with field tokenization and no `uri`; an
object holds `pos: "3"`; `verify` compares with the edited plan and passes.

*Implemented: `tests/acceptance/chroma/test_plan_edits.py::test_a_hand_edited_plan_is_copied_as_edited`. Coverage.*

**A-14 Names the server refuses pass `check`.** Level API, risk R3 and R5,
4 s. Edit the tiny.l2 draft: property `position` renamed `has space`; in a
second directory the target renamed `tiny-l2`. Expected today (suspected):
`check` exits 0 and `copy` exits 2 `target_rejected_schema` with the
server's message and the hint about hfresh, which does not apply. Expected
after a fix: `check` and `copy` exit 2 `invalid_plan` with a finding
(`invalid_name`) that quotes the server's rule (`^[_A-Za-z][_0-9A-Za-z]*$`
for properties, `^[A-Z][_0-9A-Za-z]*$` for the stored collection name).

*Implemented: `tests/acceptance/chroma/test_plan_edits.py::test_a_name_that_weaviate_refuses_is_an_invalid_plan`. Found the bug; check and copy give `invalid_name` (ruling R57).*

**A-15 `--consistency` and `--parallelism` reach the server.** Level API,
risk R3, 16 s. Parametrized: `copy --consistency ALL` and `QUORUM` on
tiny.l2 (10 objects); `copy --parallelism 1` and `--parallelism 8` on
docs-cosine (700 objects, then `verify`). Expected: exit 0 each, the
counts, `verify` exit 0. Coverage test on a single node.

*Implemented: `tests/acceptance/chroma/test_copy_options.py::test_copy_with_a_consistency_level`, `::test_copy_and_verify_at_the_bounds_of_their_options`. Coverage.*

**A-16 `verify --sample` at its bounds.** Level API, risk R3, 6 s. After a
copy of docs-cosine: `verify --sample 1` and `--sample 10000`. Expected:
exit 0; records detail `sampled 1, compared 1, ...` and `sampled 700,
compared 700, ...`.

*Implemented: `tests/acceptance/chroma/test_copy_options.py::test_copy_and_verify_at_the_bounds_of_their_options`. Coverage.*

### R4: exit codes and JSON documents

**A-17 JSON invariants on every run.** Level API, risk R4, 0 s. Test change
to `tests/e2e.py::run`: when `status` is `error`, the document has the keys
`status`, `code`, `message`, `hint`, `code` is in `ERROR_EXIT`, the exit
code equals `ERROR_EXIT[code]`, `message` is one line of at most 500
characters; when `status` is `success`, the exit code is 0; stderr holds
no JSON object. Expected: passes at once on the current suite.

*Implemented: `tests/e2e.py::run` (`_contract_problems`). Coverage.*

**A-18 Usage errors.** Level API, risk R4, 5 s. `inspect --database nope`;
`inspect --collection nope`; a second `inspect` without `--force`; `copy
--collection nope`; `copy --parallelism 9`. Expected: exit 2 `usage` each,
messages `Chroma tenant default_tenant has no collection in database
nope.`, `Chroma has no collection named nope.`, `<dir>/plan.yaml exists.`,
`--collection nope matches no plan entry.`, `--parallelism 9 is outside 1
to 8.`; no `inspect.json` written by the first two.

*Implemented: `tests/acceptance/chroma/test_usage.py::test_usage_errors_exit_2_and_name_the_value`. Coverage.*

**A-19 Text mode.** Level API, risk R4, 3 s. A helper `run_text` without
`--json`. Steps after an inspect of tiny.l2: `status`; `check` with
`WEAVIATE_API_KEY=wrong`. Expected: `status` exit 0, stdout has a line per
entry and `Total: 10 records`, stdout is not JSON; `check` exit 3, stdout
empty, stderr has `Error: Weaviate refused the request: the API key is
missing or wrong.` and a `Hint:` line naming `WEAVIATE_API_KEY`.

*Implemented: `tests/acceptance/chroma/test_usage.py::test_text_mode_prints_lines_and_the_error_on_stderr`. Coverage. In text mode the progress line goes to stdout, so stdout of the failed check holds that line.*

**U-12 A corrupted state file and a state of a newer schema.** Level unit,
risk R4. (a) `state.sqlite` holding random bytes: `status` and `copy`.
Expected today: exit 5 `internal_error` (`file is not a database`). The
plan does not call this a bug; the implementer records it for a ruling
(exit 2 with a hint that names the file would fit, like the deferred
`verify --dir <file>` minor). (b) A state file with an extra table and an
extra nullable column in `partitions`, written by a test: `copy` resumes
and `status` reads it. Expected: exit 0, the extra column untouched.

*Implemented: `tests/unit/test_cli_commands.py::test_a_corrupted_state_file_is_an_invalid_state`, `::test_a_state_file_of_a_newer_schema_resumes`. Found the bug; ruling R55 gives the code `invalid_state`.*

**U-13 The tenant from the key (`_chroma_source`).** Level unit, risk R4.
`monkeypatch` `to_weaviate.cli.ChromaClient` to pass the fake server's
transport. Cases: `CHROMA_TENANT` set: no `/auth/identity` request;
unset: one request with header `x-chroma-token: <key>`, the tenant from
the answer used in the collection paths; `CHROMA_TOKEN_HEADER=
Authorization`: header `Authorization: Bearer <key>`; identity answers
401: `inspect` exits 3 `auth_failed`, hint names `CHROMA_API_KEY`, the key
not in the output.

*Implemented: `tests/unit/test_cli_chroma.py::test_inspect_reads_the_tenant_from_the_key_only_without_chroma_tenant`, `::test_a_key_that_the_identity_refuses_exits_3`. Coverage.*

**U-14 Tenant, database and collection ids are percent-encoded in paths.**
Level unit, risk R4. Tenant `t x`, database `my db/1`, collection id with a
`+`: the fake server records the raw request paths. Expected: `/tenants/
t%20x/databases/my%20db%2F1/collections/...`, and the listing, count, get
and query all reach the right handler.

*Implemented: `tests/unit/test_cli_chroma.py::test_tenant_database_and_collection_id_are_percent_encoded_in_paths`. Coverage.*

### R5: usability

**U-15 A property added to the plan after the collection exists.** Level unit,
risk R5. Steps: `copy` of 10 records; grow the source by 5 records that
carry a new key `lang`; `copy` (exit 1, 5 rejects `metadata key lang is not
in plan.yaml`); `inspect --force` (the draft now has `lang`); `copy`.
Expected today: exit 2 `plan_mismatch`, `property lang: missing in the
collection`, hint `Edit plan.yaml to match it, or choose another target
name.` The only edit that proceeds is to list `lang` under `dropped`,
which loses the value. The case pins this and records a ruling request:
`copy` could add a planned property that the collection lacks (Weaviate
allows adding properties), or the hint could say so.

*Implemented: `tests/unit/test_cli_commands.py::test_copy_adds_a_planned_property_that_the_collection_lacks`. Found the bug; ruling R54: copy adds the property and check warns `missing_property`. The records rejected by the run before are not read again; a copy in a new directory writes them.*

**U-16 A source without vectors is drafted in keep mode without a note.**
Level unit, risk R5. `FakeSource` records with `vector=None` and no
`dimension`. Expected today: `inspect` exits 0, the entry has one `keep`
vector and no note about vectors; `copy` rejects every record with `no
vector`. Expected after a fix: a note or decision `the source has no
vectors; use --vectors embed` on the entry. Suspected usability bug; A-03
is the live twin.

*Implemented: `tests/unit/test_cli_commands.py::test_inspect_notes_and_check_refuses_a_source_without_vectors`. Found the bug; ruling R56: a note in inspect and the error `no_source_vectors` in check and copy.*

**U-17 Hints for two Chroma answers.** Level unit, risk R5. (a) 403 on the
database listing (a database-scoped Chroma Cloud key): today `auth_failed`
with the hint `Check CHROMA_API_KEY ...`; the hint should also name
`CHROMA_DATABASE` and `--database`. (b) 404 on a collection `get` at copy
time (the source collection was deleted and created again, so its id
changed): today `source_unreachable` with the hint about `CHROMA_URL`; the
hint should say `Run to-weaviate inspect chroma --force`. Both pin today's
text and propose the new one; small, last.

*Implemented: `tests/unit/test_cli_chroma.py::test_a_403_on_the_database_list_names_the_database_settings`, `::test_a_source_collection_that_is_gone_names_inspect_force`. Found the bug; both hints now name the fix (ruling R57).*

### Infra

These cases carry the marker `infra`: they need servers that the suite
owns. A-20 to A-22 need the `cluster` leg of CI: 3 Weaviate nodes,
`REPLICATION_MINIMUM_FACTOR=3`. The suite reads the node count from the
server; CI sets `ACCEPTANCE_NODES=3` to document the leg, and the suite
reports a value that differs from the server's count. The cases skip when
the cluster has fewer than 3 nodes, so they do not run locally. A-20 and A-22
delete a pod. A-23 stops the compose Chroma. After each pod deletion the case waits until
`weaviate-1` is replaced, then for `kubectl rollout status sts/weaviate -n
weaviate --timeout=60s`, then until a client connects through the
forwards, before the next command. Each fault case fails when the command
ended before the fault.

**A-20 A Weaviate pod deleted during `copy`.** Level infra, risk R1.
Parametrized: `--consistency QUORUM` and `ALL`. Steps: `inspect chroma
--collection resume --vectors both` with the suite's vectorizer, so the
copy of 8,000 records still runs after 3 s; `copy --consistency <level>`;
3 s in, `kubectl delete pod weaviate-1 -n weaviate --wait=false`; `copy
--consistency <level>` again; `verify`. Expected: the first `copy` exits
0 with `copied 8000, rejects 0, done`; `Resume` has replication factor 3;
the second `copy` exits 0 and writes no page (a `collection_finished`
event and no `progress` event in `events.jsonl`); 8,000 objects; the full
`verify` exits 0 with the `count`, `ids` and `records` checks. With
QUORUM, each object still gets two acks while one of the three replicas
is gone, so the QUORUM case may see no refused object. With ALL, a write
needs all three replicas. So the ALL case is the one that reaches the
re-send of refused objects, which PR 25 extended to a batch refused in
part. The tool writes no event for a re-send. The CI run shows that both
cases pass, not which path ran.

*Implemented: `tests/acceptance/chroma/test_cluster.py::test_a_pod_deleted_during_copy_leaves_no_reject`. Coverage.*

**A-21 Each consistency level on 3 replicas.** Level infra, risks R1 and
R3. Parametrized: `--consistency ONE`, `QUORUM` and `ALL`. Steps: `inspect
chroma --collection resume`; `copy --consistency <level>` into a new
`Resume`; `verify`. Expected: `copy` exits 0; `Resume` has replication
factor 3; 8,000 objects; `verify` exits 0.

*Implemented: `tests/acceptance/chroma/test_cluster.py::test_copy_with_a_consistency_level_on_three_replicas`. Coverage.*

**A-22 A Weaviate pod deleted during `verify`.** Level infra, risk R1.
Steps: `inspect chroma --collection bulk --sample-pages 5`; `copy
--parallelism 8`; `verify`; 3 s in, `kubectl delete pod weaviate-1 -n
weaviate --wait=false`. A full `verify` of 50,000 records took 4.96 s and
4.15 s on one local node, so it still runs at 3 s there. Expected:
`verify` exits 0 with the `count`, `ids` and `records` checks passed.

*Implemented: `tests/acceptance/chroma/test_cluster.py::test_verify_passes_while_a_pod_is_deleted`. Coverage.*

**A-23 Chroma stopped and started during `copy`.** Level infra, risk R1,
30 s. Steps: `inspect chroma --collection resume --vectors both` with the
suite's vectorizer; `copy`; 2 s in, `docker compose -f
tests/acceptance/chroma/docker-compose.yml stop`, 3 s of wait, then
`docker compose ... start`; the case waits until Chroma answers with the
8,000 records of `resume`; `verify`. Expected: `copy` exits 0 with
`copied 8000, rejects 0, done`; 8,000 objects; `verify` exits 0. Runs on
the `single` leg and locally. It skips with `ACCEPTANCE_CHROMA=distributed`,
because the `cluster` leg runs no compose Chroma. Locally Chroma answered
no request for about 3.7 s. The tool writes no event for a source retry,
so the case cannot show that one ran.

*Implemented: `tests/acceptance/chroma/test_chroma_restart.py::test_a_chroma_restart_during_copy_leaves_no_reject`. Coverage.*

### Distributed Chroma

`copy` reads by offset: the partitions, the resume and the shift detection
(rulings R61 and R65) need `get` with `offset` and `limit` to return the
records in insertion order. A-24 pins that order. A-25 copies right after
a burst of writes, and the compaction guard (U-18) makes the copy wait.
The `cluster` leg of CI runs the suite against the open-source distributed
Chroma, which `tests/acceptance/chroma-k8s.sh` starts in its kind cluster,
with `ACCEPTANCE_CHROMA=distributed`. When a rollout fails, the script
prints the runner's CPU and the last exit code of each pod that is not
ready; exit code 132 (SIGILL) means that the runner's CPU lacks an
instruction set of the query and compaction images, and a rerun of the
failed job on another runner passes. On that leg the seed waits until
`indexing_status` reports `num_unindexed_ops` 0 for each seeded
collection, so the other cases read compacted collections. Distributed
Chroma compacts every 10 s and takes at most 10,000 records of a
collection per run. A run of more than 5,000 records gives them new
offsets, in another order. Its frontend keeps a collection for 60 s, and
`get` and `indexing_status` read the collection from that cache.
Distributed Chroma makes a SPANN index where single-node Chroma makes
HNSW, so on that leg every case, the distances case too, reads SPANN
collections.

**A-24 `get` pages follow the insertion order.** Level API, risk R1, 1 s
locally. Steps: make `order` with 1,000 records of 4 dimensions in adds of
300; their UUID ids do not sort in insertion order. Read every page of 300
from offset 0. Delete the records at positions 400, 450, 500, 550 and 600,
and read again. Add 10 records, and read again. On distributed Chroma the
case waits for `num_unindexed_ops` 0 after each write; each wait can take
more than 60 s because of the frontend's cache. Expected: the ids in
insertion order. After the deletion, the 5 ids are gone, and each later
record is one offset earlier per deleted record before it. After the
addition, the 10 ids come last. Runs on both Chroma kinds.

*Implemented: `tests/acceptance/chroma/test_source_order.py::test_get_pages_follow_the_insertion_order`. Coverage.*

**U-18 The compaction guard.** Level unit, risk R1. The client sends
`GET .../collections/{id}/indexing_status`. A 500 whose message holds "is
not implemented", the answer of single-node Chroma 1.5.9, gives None
without a retry; any other 500 is retried as before (ruling R99). A 404,
the answer of a server before Chroma 1.4.1, gives None too (ruling R103).
The client reads the collection document by name, from sysdb. The source
sends one `indexing_status` and returns None on single-node Chroma: one
request, no wait. Otherwise each poll reads the collection's `version` and
`log_position` first and `indexing_status` second, every 5 s, and returns
the state read before the first 0. It reports each change of
`num_unindexed_ops` and raises `SourceNotCompacted` after 600 s (ruling
R97). A collection document with another id than the inspected one gives
the hint of a collection created again. The engine waits before
`init_collection`, writes the event `compaction_wait` for each report, and
reads the state again after the last read. A changed state adds a tail
from offset 0 that supersedes every stored partition and raises
`SourceChanged`, so the next run reads the collection again; a source
without compaction is not compared. `verify` waits before its counts and
compares after its records, and a stop writes no report. Both stops are
`source_compacting`, exit 4, with their own message and hint (ruling R98).

*Implemented: `tests/unit/test_chroma_client.py::test_indexing_status_returns_the_counts`, `::test_indexing_status_of_a_single_node_server_is_none_and_not_retried`, `::test_another_500_on_indexing_status_is_retried`, `::test_collection_by_name_reads_the_collection_document`; `tests/unit/test_chroma_source.py::test_wait_compacted_returns_the_state_when_nothing_waits`, `::test_wait_compacted_polls_until_nothing_waits`, `::test_wait_compacted_raises_after_600_seconds`, `::test_wait_compacted_on_single_node_sends_one_request`, `::test_compaction_state_of_a_collection_created_again_names_inspect_force`; `tests/unit/test_copy.py::test_the_wait_comes_before_the_first_page_and_the_state_after_the_last_read`, `::test_a_changed_state_stops_the_copy_and_the_next_run_reads_from_the_start`, `::test_a_source_without_compaction_is_not_compared`, `::test_the_wait_writes_a_compaction_wait_event_per_change`, `::test_a_wait_that_ends_without_compaction_reads_no_page`; `tests/unit/test_verify.py::test_verify_waits_before_the_counts_and_reads_the_state_after_the_records`, `::test_verify_stops_with_source_compacting_and_writes_no_report`; `tests/unit/test_cli_json.py::test_a_compaction_during_copy_is_source_compacting_exit_4`, `::test_a_collection_that_chroma_does_not_compact_is_source_compacting_exit_4`; `tests/unit/test_cli_chroma.py::test_copy_and_verify_wait_until_chroma_compacted`, `::test_a_single_node_chroma_gets_one_indexing_status_request_per_command`. Found by A-25; ruling R70.*

**A-25 `copy` while distributed Chroma compacts.** Level API, risk R1,
distributed Chroma only. Steps: make `burst` with 40,000 records of 16
dimensions in adds of 300; `inspect chroma --collection burst
--sample-pages 1`; read the pages at offsets 0, 5,000, ... 35,000 and the
collection's `version` and `log_position`; `copy --parallelism 8` right
after, without the suite's wait for compaction; read the same pages and
values again; the full `verify`. Expected: `copy` first waits for
`num_unindexed_ops` 0, then exits 0, and `verify` gives `source count
40000, target count 40000` and `missing 0, extra 0`. The
frontend serves `get` from a collection cache whose entries live 60 s, so
a renumbering becomes visible at a cache expiry, not when the compaction
run ends. The case records `crossed` (page 0 changed during the copy),
`changed_offsets` (the read pages that changed), both states and the
durations as test properties in the JUnit XML. On PR #34 CI run 2, with
the compactor at 200m CPU, the copy had `crossed` true and
`changed_offsets` [0, 5000]. `version` went from 0 to 1 and `log_position`
from 0 to 10000 during the copy. `verify` found `source count 40000,
target count 38631` and `missing 1369, extra 0`. So a copy across a
renumbering skips records, the tool bug of ruling R70. Its fix (U-18):
`copy` waits for `num_unindexed_ops` 0 before the first page, and stops
with `source_compacting` when the collection's `version` or
`log_position` changed before the last page.

*Implemented: `tests/acceptance/chroma/test_source_order.py::test_copy_during_compaction`. Found the bug on PR #34 CI run 2; the guard of ruling R70 fixes it (U-18).*

### Cloud

**L-01 Local Chroma to Weaviate Cloud, API level, run by hand before a
release.** Level local cloud run, risks R1 to R4. Preconditions: the
local test Chroma (`make chroma-up`), a dedicated serverless Weaviate
Cloud cluster whose `/v1/meta` lists `text2vec-weaviate`, and
`WEAVIATE_URL` and `WEAVIATE_API_KEY` of that cluster in the environment.
Steps: `make acceptance-cloud`. It runs every API-level test of
`tests/acceptance` against the cluster, with `ACCEPTANCE_TARGET=cloud`,
`WEAVIATE_ASYNC_INDEXING=true` and `ACCEPTANCE_VECTORIZER=text2vec-weaviate`.
Expected: the results of the CI run with `ASYNC_INDEXING`, without the
`infra` tests, which the cloud run deselects. The release
notes record the passed, skipped and failed tests, with the reasons. L-01
replaces the deleted Chroma Cloud suite `tests/cloud` and its cases C-02
and C-03, because no Chroma Cloud source exists.

*Implemented: the `acceptance-cloud` target of the `Makefile`.*

## 5. What cannot be tested here

- Chroma Cloud: no Chroma Cloud account exists, so no test reads from it.
  The unit tests cover the tool's side with fakes, and L-01 covers the
  Weaviate Cloud side. A-24 and A-25 cover the read order and the
  compaction of the open-source distributed Chroma. These stay unverified:
  - The read order of a sharded collection. Distributed Chroma reads it
    shard by shard, which is not insertion order. Sharding is off by
    default and turned on per tenant pattern (`compaction_manager.rs`), so
    it can be on in Chroma Cloud. The CI cluster has no sharded collection.
  - The `max_partition_size` of Chroma Cloud. The open-source worker config
    sets 5,000, the run size above which compaction gives records new
    offsets.
  - Whether Chroma Cloud answers `indexing_status` as the open-source
    frontend does, and how long its compaction takes. The 600 s bound of
    the compaction guard comes from the CI cluster.
  - A compaction between two runs. `copy` compares the state only within
    a run (follow-up: store the start state in `state.sqlite`). Until then
    the full `verify` after the last copy finds the skipped records.
  - Quotas and the 429 behavior: the page cap of 300 results per request
    (whether a request above 300 records is refused or cut, which decides
    whether U-01 is a loss in practice), the limit of 10 concurrent reads
    per collection (`copy --parallelism` above 10), the rate limits and the
    `Retry-After` values. The open-source frontend enforces no quota.
  - Collections with sparse vectors.
  - The shapes of the stored embedding-function configuration.
  - Chroma Cloud authentication, live: the API key, the scope of a
    database-scoped key and the `identity` answer.
- A Weaviate Cloud sandbox: the cloud run (L-01) uses a serverless
  cluster. A sandbox refuses `hnsw`, `flat`, `dynamic` and the quantizer
  settings, so those cases fail there with the server's message. No test
  runs a copy that a sandbox accepts. Manual check of the `SCHEMA_HINT`
  advice (hfresh): `inspect --index hfresh --quantizer rq
  --quantizer-bits 1`, `check`, `copy` against a sandbox.
- Other Weaviate versions: the rules of B8 bugs 1 to 3 and 8, the reserved
  suffixes and the sidecar shape are 1.40.4 behavior (older servers accept
  more, so `check` refuses plans they would take). `tests/acceptance/
  weaviate-version` pins the CI version; a version bump reruns the suite.
- A server without `ASYNC_INDEXING`: both CI legs run with it, as every
  Weaviate Cloud instance does. Without it, the update of a trained
  quantizer after the pages (R50) compresses the vectors at once, and
  Weaviate refuses a `dynamic` index. Unit tests cover the tool's side:
  `cmd::test_copy_enables_a_quantizer_that_needs_training_after_the_pages`,
  `cmd::test_a_refused_quantizer_update_stops_copy_and_resumes_without_pages`
  and `cmd::test_check_warns_that_a_dynamic_index_needs_async_indexing`.
  The acceptance suite runs against such a server locally:
  `ASYNC=0 make weaviate-up`, then
  `WEAVIATE_ASYNC_INDEXING=false make acceptance`.
- RBAC with a key that reads but cannot write: the CI action creates
  `admin-key` only. U-04 covers the error path with fakes. A live check
  needs `auth-config` in `ci.yml` (the local-k8s `rbac.yaml` example adds
  `viewer-key`) and the local container recreated, which this batch does
  not do. Manual check before a release: assign the `viewer` role to a
  second key, run `check` (exit 0) and `copy` (exit 3, nothing created).
- Faults that need a proxy: a connection reset in the middle of a response
  body, a slow server that trips the 60 s Chroma timeout or the client's
  timeouts, a server that answers after a partial write. Unit cases U-02
  and the writer tests inject them; an acceptance version needs a proxy
  such as toxiproxy in the compose file (a later batch).
- Inputs Chroma 1.5.9 refuses, so no source holds them: vectors of mixed
  dimension in one collection, a NaN, an infinity or a null inside an
  embedding, an int outside int64, a unicode collection name. Unit cases
  cover the tool's side (U-05, the existing NaN tests).
- A re-ingest with the same ids between two copies (R66): the last two
  records read, deleted and added back in their old order with records
  added in between, hide those records from the next `copy`; only the full
  `verify` ids check finds them, and `sync` (Milestone 2) fixes it.
  On distributed Chroma, a record that is deleted and added again before
  the next compaction run keeps its old offset, so such a re-ingest hides
  no record there. A record added again after the run that compacted its
  deletion goes last, as on single-node Chroma. Both facts come from
  Chroma's code; no case tests them.
- Windows (`copy` without the lock), Ctrl-C (deferred F39), a full disk,
  a read-only migration directory, a source of millions of records
  (`inspect` without `--sample-pages` reads every page; manual check: a
  1M-record copy with `--sample-pages 20`, timing of each command, and a
  kill and resume in the middle).
- Provider vectorizers (`text2vec-openai` and the others): Weaviate
  1.40.4 enables every API-based module by default, but the CI server has
  no provider key and serves inference for `text2vec-transformers` only.
  A-09 covers the module check with the first of `CANDIDATE_MODULES`
  that `/v1/meta` does not list; the
  mapping tests cover the options. Manual check before a release: one
  embed-mode copy with `text2vec-openai` and a real key, and a read of
  `rejects.jsonl` after a wrong key (the provider's message may echo part
  of the key; deferred F26).
