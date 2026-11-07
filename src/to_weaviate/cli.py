import warnings

# authlib comes with weaviate-client and is not a declared dependency.
try:
    from authlib.deprecate import AuthlibDeprecationWarning
except ImportError:
    pass
else:
    # Importing authlib.deprecate adds an "always" filter for this category, so
    # this filter comes after that import. It must come before weaviate is
    # imported.
    warnings.filterwarnings(
        "ignore",
        message="The httpx module is deprecated",
        category=AuthlibDeprecationWarning,
    )

import atexit
import dataclasses
import json
import os
import random
import re
import sys
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
import typer
import weaviate
import yaml
from pydantic import ValidationError
from weaviate.exceptions import (
    AuthenticationFailedError,
    InsufficientPermissionsError,
    UnexpectedStatusCodeError,
    WeaviateBaseError,
    WeaviateConnectionError,
    WeaviateGRPCUnavailableError,
    WeaviateStartUpError,
    WeaviateTimeoutError,
)

from to_weaviate import __version__
from to_weaviate.engine.check import (
    Decision,
    Finding,
    collection_key,
    decisions,
    load_findings,
    module_findings,
    plan_findings,
    provider_keys,
    target_groups,
    userinfo_only,
)
from to_weaviate.engine.copy import copy_collection
from to_weaviate.engine.verify import (
    REPORT_MD,
    Check,
    check_line,
    diff_ids,
    missing_target,
    remove_report,
    target_label,
    verify_counts,
    verify_ids,
    verify_records,
    write_report,
)
from to_weaviate.events import Events
from to_weaviate.lock import Locked, exclusive_lock
from to_weaviate.output import (
    ERROR_EXIT,
    CliError,
    emit,
    info,
    redact,
    redact_strings,
    secret_values,
)
from to_weaviate.plan import (
    CollectionPlan,
    IndexPlan,
    Plan,
    SourcePlan,
    VectorizerPlan,
    draft_plan,
    load_plan,
    save_plan,
)
from to_weaviate.settings import (
    ChromaSettings,
    InvalidSetting,
    MissingCredential,
    WeaviateSettings,
    chroma_settings,
    load_env,
    weaviate_settings,
    without_userinfo,
)
from to_weaviate.sources.base import FieldStats, Source, SourceCollection
from to_weaviate.sources.chroma.client import ChromaClient, ChromaError
from to_weaviate.sources.chroma.mapping import stored_embedding, vectorizer_for
from to_weaviate.sources.chroma.source import ChromaSource
from to_weaviate.state import CollectionProgress, State, Target
from to_weaviate.target.connect import connect
from to_weaviate.target.reader import TargetReader, WeaviateReader
from to_weaviate.target.schema import (
    PlanMismatch,
    Schema,
    SchemaRejected,
    WeaviateSchema,
)
from to_weaviate.target.writer import BatchRefused, WeaviateWriter, Writer

SourceFactory = Callable[[ChromaSettings], Source]
WriterFactory = Callable[[WeaviateSettings, str | None], Writer]
ReaderFactory = Callable[[WeaviateSettings], TargetReader]
SchemaFactory = Callable[[WeaviateSettings], Schema]

VectorsMode = Literal["keep", "embed", "both"]
IndexType = Literal["default", "hnsw", "flat", "dynamic", "hfresh"]
QuantizerType = Literal["default", "none", "rq", "sq", "pq", "bq"]
Consistency = Literal["ONE", "QUORUM", "ALL"]

DEFAULT_DIR = Path(".to-weaviate")
INSPECT_FILE = "inspect.json"
PLAN_FILE = "plan.yaml"
STATE_FILE = "state.sqlite"
LOCK_FILE = "copy.lock"
EVENTS_FILE = "events.jsonl"
REJECTS_FILE = "rejects.jsonl"
DEFAULT_PARALLELISM = 5
MAX_PARALLELISM = 8
DEFAULT_SAMPLE = 300
MAX_SAMPLE = 10_000
MAX_MESSAGE_CHARS = 500

INSPECT_HINT = "Run to-weaviate inspect chroma to create it."
REINSPECT_HINT = "Run to-weaviate inspect chroma --force to inspect again."
EDIT_PLAN_HINT = (
    "Edit plan.yaml, or run to-weaviate inspect chroma --force to draft it again."
)
USERINFO_HINT = "Percent-encode the password in plan.yaml or remove the option."
UTF8_HINT = (
    "Save the file as UTF-8, or run to-weaviate inspect chroma --force to write "
    "it again."
)
CHROMA_HINT = "Check CHROMA_URL and that the Chroma server is running."
NO_COLLECTION_HINT = "Check --database, CHROMA_DATABASE and CHROMA_TENANT."
WEAVIATE_HINT = (
    "Check WEAVIATE_URL, WEAVIATE_GRPC_URL and that the Weaviate server is running."
)
ISSUES_URL = "https://github.com/weaviate/to-weaviate/issues"
INTERNAL_HINT = (
    f"Report it at {ISSUES_URL}. Set TO_WEAVIATE_DEBUG=1 to print the traceback."
)
DEBUG_VARIABLE = "TO_WEAVIATE_DEBUG"
SCHEMA_HINT = (
    "A free Weaviate Cloud cluster accepts only the hfresh index. Run "
    "to-weaviate inspect chroma --index hfresh --force, or edit plan.yaml."
)
STATE_HINT = (
    "To resume, set WEAVIATE_URL and the target name back. To copy to the new "
    "target, use another --dir."
)
BATCH_HINT = "Fix the cause and run to-weaviate copy again. It resumes at this page."
# A part of a text in single or double quotes; a quote after a backslash is
# part of it, as in a Python repr.
_QUOTED = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")


@dataclass(frozen=True)
class _Factories:
    source: SourceFactory
    writer: WriterFactory
    reader: ReaderFactory
    schema: SchemaFactory


def _short(text: str) -> str:
    """One line of at most MAX_MESSAGE_CHARS characters."""
    line = " ".join(text.split())
    if len(line) <= MAX_MESSAGE_CHARS:
        return line
    return line[: MAX_MESSAGE_CHARS - 3] + "..."


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _validation_text(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error["loc"])
        parts.append(f"{loc}: {error['msg']}" if loc else error["msg"])
    return "; ".join(parts)


def _refused(service: str, key: str, forbidden: bool) -> CliError:
    reason = (
        "the API key lacks a permission"
        if forbidden
        else "the API key is missing or wrong"
    )
    return CliError(
        "auth_failed",
        f"{service} refused the request: {reason}.",
        f"Check {key} in the environment or in .env.",
    )


def _as_cli_error(exc: Exception) -> CliError:
    """The CliError for `exc`. An exception outside the table is internal_error.

    Authentication messages do not repeat the server's text, because a server
    can echo the credential it refused.
    """
    if isinstance(exc, CliError):
        return exc
    if isinstance(exc, MissingCredential):
        return CliError(
            "missing_credential",
            str(exc),
            f"Set {exc.var} in the environment or in .env.",
        )
    if isinstance(exc, InvalidSetting):
        return CliError("usage", str(exc), exc.hint)
    if isinstance(exc, ChromaError):
        if exc.status in (401, 403):
            return _refused("Chroma", "CHROMA_API_KEY", exc.status == 403)
        if exc.status is None:
            return CliError(
                "source_unreachable", f"Chroma could not be reached: {exc}", CHROMA_HINT
            )
        return CliError("source_unreachable", str(exc), CHROMA_HINT)
    if isinstance(exc, httpx.TransportError):
        # ChromaClient wraps its own transport errors in ChromaError, so a raw
        # one comes from the Weaviate client.
        return CliError(
            "target_unreachable",
            f"Weaviate could not be reached: {type(exc).__name__}: {exc}",
            WEAVIATE_HINT,
        )
    if isinstance(exc, AuthenticationFailedError):
        return _refused("Weaviate", "WEAVIATE_API_KEY", False)
    if isinstance(exc, InsufficientPermissionsError):
        return _refused("Weaviate", "WEAVIATE_API_KEY", True)
    if isinstance(exc, UnexpectedStatusCodeError) and exc.status_code in (401, 403):
        return _refused("Weaviate", "WEAVIATE_API_KEY", exc.status_code == 403)
    if isinstance(
        exc,
        WeaviateConnectionError
        | WeaviateStartUpError
        | WeaviateTimeoutError
        | WeaviateGRPCUnavailableError,
    ):
        return CliError(
            "target_unreachable",
            f"Weaviate could not be reached: {exc}",
            WEAVIATE_HINT,
        )
    if isinstance(exc, SchemaRejected):
        return CliError(
            "target_rejected_schema",
            f"Weaviate refused to create the collection: {exc}",
            SCHEMA_HINT,
        )
    if isinstance(exc, BatchRefused):
        return CliError("target_unreachable", str(exc), BATCH_HINT)
    if isinstance(exc, PlanMismatch):
        return CliError(
            "plan_mismatch",
            str(exc),
            "The target collection exists with other settings. Edit plan.yaml "
            "to match it, or choose another target name.",
        )
    if isinstance(exc, WeaviateBaseError):
        return CliError(
            "target_unreachable",
            f"Weaviate request failed: {type(exc).__name__}: {exc}",
            WEAVIATE_HINT,
        )
    return CliError(
        "internal_error",
        f"internal error: {type(exc).__name__}: {exc}",
        INTERNAL_HINT,
    )


def _handler_env() -> dict[str, str]:
    # The handler must report the original error, so a `.env` that cannot be
    # read falls back to the process environment.
    try:
        return load_env(Path.cwd())
    except Exception:
        return dict(os.environ)


@dataclass(frozen=True)
class _Report:
    exit_code: int
    document: dict[str, Any]
    text: str
    trace: str


_FALLBACK_REPORT = _Report(
    ERROR_EXIT["internal_error"],
    CliError(
        "internal_error",
        "internal error: the error could not be redacted, so its details are "
        "not printed",
        f"Report it at {ISSUES_URL}.",
    ).document(),
    "",
    "",
)


def _report(exc: Exception) -> _Report:
    """The redacted output for `exc`. Builds strings only; prints nothing."""
    env = _handler_env()
    secrets = secret_values(env)
    error = _as_cli_error(exc)
    trace = ""
    if env.get(DEBUG_VARIABLE, "").strip() == "1":
        lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
        trace = redact("".join(lines), secrets)
    document = redact_strings(error.document(), secrets)
    # Shortening comes after redaction: a cut or a collapsed space would
    # change a secret so that it no longer matches.
    document["message"] = _short(document["message"])
    return _Report(error.exit_code, document, redact(error.text, secrets), trace)


@contextmanager
def _handled(json_output: bool) -> Iterator[None]:
    """Report a failure as text or as a JSON document, then exit.

    Typer and Click exits keep their own behavior. Secret values are redacted
    from everything printed here.
    """
    try:
        yield
    except (typer.Exit, typer.Abort, typer.TyperException):
        raise
    except Exception as exc:
        try:
            report = _report(exc)
        except Exception:
            # Nothing from either exception is printed: it could not be redacted.
            report = _FALLBACK_REPORT
        if report.trace:
            print(report.trace, end="", file=sys.stderr)
        emit(json_output, report.document, report.text)
        if not json_output:
            print(f"Error: {report.document['message']}", file=sys.stderr)
            print(f"Hint: {report.document['hint']}", file=sys.stderr)
        raise typer.Exit(report.exit_code) from None


def _yaml_error_text(exc: yaml.YAMLError) -> str:
    """The position and the problem of `exc`, without any text of the file.

    The text that PyYAML gives for an error quotes the line, which can hold a
    password. Its problem text can quote a name from the line, such as an
    alias, a tag or a key, so each quoted part becomes "...".
    """
    if isinstance(exc, yaml.MarkedYAMLError) and exc.problem_mark is not None:
        mark = exc.problem_mark
        where = f"YAML error at line {mark.line + 1}, column {mark.column + 1}"
        if not exc.problem:
            return where
        return f"{where}: {_QUOTED.sub('...', exc.problem)}"
    return "the text is not valid YAML"


def _not_utf8(path: Path) -> CliError:
    return CliError("invalid_plan", f"{path} is not UTF-8 text.", UTF8_HINT)


def _read_plan(directory: Path) -> Plan:
    """The plan, parsed and validated, before `load_findings` are applied."""
    path = directory / PLAN_FILE
    if not path.exists():
        raise CliError("usage", f"{path} does not exist.", INSPECT_HINT)
    try:
        plan = load_plan(path)
    except UnicodeDecodeError as exc:
        raise _not_utf8(path) from exc
    except ValidationError as exc:
        raise CliError(
            "invalid_plan",
            f"{path} is not a valid plan: {_validation_text(exc)}",
            EDIT_PLAN_HINT,
        ) from exc
    except yaml.YAMLError as exc:
        # Not chained: the YAML error holds the line, and the debug traceback
        # would print it.
        raise CliError(
            "invalid_plan",
            f"{path} is not a valid plan: {_yaml_error_text(exc)}",
            EDIT_PLAN_HINT,
        ) from None
    if not plan.collections:
        raise CliError(
            "invalid_plan",
            f"{path} is not a valid plan: collections is empty.",
            EDIT_PLAN_HINT,
        )
    return plan


def _load_plan(directory: Path) -> Plan:
    """The plan, refused when it has a `load_findings` error."""
    plan = _read_plan(directory)
    errors = load_findings(plan)
    if errors:
        raise _plan_errors(errors, [])
    return plan


def _source_collection(raw: dict[str, Any]) -> SourceCollection:
    fields = [FieldStats(**f) for f in raw["fields"]]
    return SourceCollection(**{**raw, "fields": fields})


def _load_inspected(directory: Path) -> dict[str, SourceCollection] | None:
    """Inspected collections by ref, or None when the file is missing."""
    path = directory / INSPECT_FILE
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        collections = [_source_collection(c) for c in raw["collections"]]
    except UnicodeDecodeError as exc:
        raise _not_utf8(path) from exc
    except (ValueError, KeyError, TypeError) as exc:
        raise CliError(
            "usage", f"{path} cannot be read: {exc}", REINSPECT_HINT
        ) from exc
    return {c.ref: c for c in collections}


def _load_plan_and_inspection(
    directory: Path,
) -> tuple[Plan, dict[str, SourceCollection]]:
    """The plan, and the inspected collections by ref, with one for every entry."""
    plan = _load_plan(directory)
    inspected = _load_inspected(directory)
    if inspected is None:
        raise CliError(
            "usage", f"{directory / INSPECT_FILE} does not exist.", INSPECT_HINT
        )
    missing = [e.source.name for e in plan.collections if e.source.ref not in inspected]
    if missing:
        raise CliError(
            "usage",
            f"{INSPECT_FILE} has no inspection of {', '.join(missing)}.",
            REINSPECT_HINT,
        )
    return plan, inspected


def _option_value(text: str) -> Any:
    if text.isascii() and text.isdigit():
        return int(text)
    if text in ("true", "false"):
        return text == "true"
    return text


def _vectorizer_plan(module: str | None, options: list[str]) -> VectorizerPlan | None:
    parsed: dict[str, Any] = {}
    for option in options:
        key, sep, value = option.partition("=")
        if not sep or not key:
            raise CliError(
                "usage",
                f"--vectorizer-option {option} is not KEY=VALUE.",
                "Write each option as --vectorizer-option KEY=VALUE.",
            )
        parsed[key] = _option_value(value)
    if module is None:
        if parsed:
            raise CliError(
                "usage",
                "--vectorizer-option needs --vectorizer.",
                "Name the module, for example --vectorizer text2vec-openai.",
            )
        return None
    return VectorizerPlan(module=module, options=parsed)


def _index_plan(
    index: IndexType, quantizer: QuantizerType, bits: int | None
) -> IndexPlan:
    if bits is not None and quantizer != "rq":
        raise CliError(
            "usage",
            f"--quantizer-bits applies only to --quantizer rq, not {quantizer}.",
            "Remove --quantizer-bits, or use --quantizer rq.",
        )
    try:
        return IndexPlan.model_validate(
            {"type": index, "quantizer": {"type": quantizer, "bits": bits}}
        )
    except ValidationError as exc:
        raise CliError(
            "usage",
            f"--index {index} --quantizer {quantizer}: {_validation_text(exc)}",
            "A quantizer needs a named --index; --quantizer-bits takes 1, 4 or 8.",
        ) from exc


def _vectors_text(entry: CollectionPlan) -> str:
    parts = []
    for v in entry.target.vectors:
        module = v.vectorizer.module if v.vectorizer else "no vectorizer"
        parts.append(f"{v.name} ({v.mode}, {module})")
    return ", ".join(parts)


def _no_collection(tenant: str, databases: list[str]) -> CliError:
    """The error of an inspection that found no collection in `databases`."""
    if databases:
        where = f"has no collection in database {', '.join(databases)}"
    else:
        where = "has no database"
    return CliError("usage", f"Chroma tenant {tenant} {where}.", NO_COLLECTION_HINT)


def _stored_collections(
    collections: list[SourceCollection],
) -> tuple[list[SourceCollection], dict[str, list[str]]]:
    """`collections` with the stored form of their embedding, and notes by ref."""
    stored: list[SourceCollection] = []
    notes: dict[str, list[str]] = {}
    for collection in collections:
        embedding, notes[collection.ref] = stored_embedding(collection.embedding)
        stored.append(dataclasses.replace(collection, embedding=embedding))
    return stored, notes


def _inspect_chroma(
    factories: _Factories,
    directory: Path,
    databases: list[str],
    names: list[str],
    sample_pages: int | None,
    vectors: VectorsMode,
    vectorizer: str | None,
    vectorizer_options: list[str],
    index: IndexType,
    quantizer: QuantizerType,
    quantizer_bits: int | None,
    force: bool,
    json_output: bool,
) -> None:
    plan_path = directory / PLAN_FILE
    if plan_path.exists() and not force:
        raise CliError(
            "usage",
            f"{plan_path} exists.",
            "Run the command again with --force to replace it.",
        )
    vectorizer_plan = _vectorizer_plan(vectorizer, vectorizer_options)
    index_plan = _index_plan(index, quantizer, quantizer_bits)
    settings = chroma_settings(load_env(Path.cwd()))
    if not databases and settings.database is not None:
        databases = [settings.database]
    source = factories.source(settings)
    info(json_output, f"Inspecting Chroma at {settings.stored_url}")
    collections = source.inspect(databases or None, names or None, sample_pages)
    missing = sorted(set(names) - {c.name for c in collections})
    if missing:
        raise CliError(
            "usage",
            f"Chroma has no collection named {', '.join(missing)}.",
            "Check the --collection names, and --database or CHROMA_DATABASE for "
            "another database.",
        )
    if not collections:
        raise _no_collection(source.tenant, databases or source.namespaces())
    collections, source_notes = _stored_collections(collections)
    try:
        plan = draft_plan(
            SourcePlan(type="chroma", url=settings.stored_url, tenant=settings.tenant),
            collections,
            vectors,
            vectorizer_plan,
            index=index_plan,
            vectorizer_for=vectorizer_for,
            source_notes=source_notes,
        )
    except ValidationError as exc:
        raise CliError(
            "usage", _validation_text(exc), "Change --index or --quantizer."
        ) from exc
    directory.mkdir(parents=True, exist_ok=True)
    inspect_document = {"collections": [dataclasses.asdict(c) for c in collections]}
    (directory / INSPECT_FILE).write_text(
        json.dumps(inspect_document, indent=2), encoding="utf-8"
    )
    save_plan(plan, plan_path)

    found = decisions(plan, {c.ref: c for c in collections})
    document, text = _inspect_output(plan, found, plan_path)
    emit(json_output, document, text)


def _inspect_output(
    plan: Plan, found: list[Decision], plan_path: Path
) -> tuple[dict[str, Any], str]:
    rows = [
        {
            "ref": e.source.ref,
            "source": e.source.name,
            "target": e.target.name,
            "count": e.source.count,
            "dimension": e.source.dimension,
            "distance": e.source.distance,
            "vectors": [
                {
                    "name": v.name,
                    "mode": v.mode,
                    "vectorizer": v.vectorizer.module if v.vectorizer else None,
                }
                for v in e.target.vectors
            ],
            "notes": e.notes,
        }
        for e in plan.collections
    ]
    lines: list[str] = []
    for e in plan.collections:
        lines.append(f"{e.source.name} -> {e.target.name}")
        lines.append(f"  count: {e.source.count}")
        lines.append(f"  dimension: {e.source.dimension}")
        lines.append(f"  distance: {e.source.distance}")
        lines.append(f"  vectors: {_vectors_text(e)}")
        lines.extend(f"  note: {note}" for note in e.notes)
    if found:
        lines.append("Decisions:")
        lines.extend(f"  {d.collection} ({d.kind}): {d.detail}" for d in found)
    lines.append(f"Review {plan_path}, then run to-weaviate check.")
    document = {
        "status": "success",
        "plan": str(plan_path),
        "collections": rows,
        "total": sum(e.source.count for e in plan.collections),
        "decisions": [dataclasses.asdict(d) for d in found],
    }
    return document, "\n".join(lines)


def _findings_text(errors: list[Finding], warnings: list[Finding]) -> list[str]:
    return [
        f"{kind} {f.code} {f.collection}: {f.message}"
        for kind, items in (("error", errors), ("warning", warnings))
        for f in items
    ]


def _findings_document(
    errors: list[Finding], warnings: list[Finding]
) -> dict[str, Any]:
    return {
        "errors": [dataclasses.asdict(f) for f in errors],
        "warnings": [dataclasses.asdict(f) for f in warnings],
    }


def _plan_errors(errors: list[Finding], warnings: list[Finding]) -> CliError:
    return CliError(
        "invalid_plan",
        f"The plan has {_plural(len(errors), 'error')}.",
        USERINFO_HINT if userinfo_only(errors) else EDIT_PLAN_HINT,
        fields=_findings_document(errors, warnings),
        text="\n".join(_findings_text(errors, warnings)),
    )


def _require_provider_keys(plan: Plan, settings: WeaviateSettings) -> None:
    for header, variable in provider_keys(plan):
        if header not in settings.headers:
            raise MissingCredential(variable)


def _check(factories: _Factories, directory: Path, json_output: bool) -> None:
    plan = _read_plan(directory)
    inspected = _load_inspected(directory)
    errors, warnings = plan_findings(plan, inspected, vectorizer_for)
    errors[:0] = load_findings(plan)
    settings = weaviate_settings(load_env(Path.cwd()))
    _require_provider_keys(plan, settings)
    info(json_output, f"Checking the target at {without_userinfo(settings.url)}")
    schema = factories.schema(settings)
    errors.extend(module_findings(plan, schema.modules()))
    for name, group in target_groups(plan.collections).items():
        for difference in schema.differences(group) or []:
            errors.append(Finding("plan_mismatch", name, difference))
    if errors:
        raise _plan_errors(errors, warnings)
    lines = _findings_text(errors, warnings)
    lines.append(f"No errors, {_plural(len(warnings), 'warning')}.")
    document = {"status": "success", **_findings_document(errors, warnings)}
    emit(json_output, document, "\n".join(lines))


def _rows(
    entries: list[CollectionPlan], progress: dict[str, CollectionProgress]
) -> list[dict[str, Any]]:
    rows = []
    for entry in entries:
        p = progress.get(entry.source.ref)
        rows.append(
            {
                "ref": entry.source.ref,
                "target": entry.target.name,
                "total": p.total if p else entry.source.count,
                "copied": p.copied if p else 0,
                "rejects": p.rejects if p else 0,
                "done": p.done if p else False,
            }
        )
    return rows


def _target_text(target: Target) -> str:
    tenant = f", tenant {target.tenant}" if target.tenant else ""
    return f"{target.host}, collection {target.name}{tenant}"


def _progress_output(
    entries: list[CollectionPlan],
    rows: list[dict[str, Any]],
    targets: dict[str, Target] | None = None,
) -> tuple[dict[str, Any], str]:
    """The progress document and text. `targets` adds each stored target to both."""
    total = sum(row["total"] for row in rows)
    lines = []
    for entry, row in zip(entries, rows, strict=True):
        lines.append(
            f"{entry.source.name} -> {row['target']}: copied {row['copied']} of "
            f"{row['total']}, rejects {row['rejects']}, "
            f"{'done' if row['done'] else 'not done'}"
        )
        target = (targets or {}).get(entry.source.ref)
        if target is not None:
            row["stored_target"] = {
                "host": target.host,
                "collection": target.name,
                "tenant": target.tenant,
            }
            lines.append(f"  stored target: {_target_text(target)}")
    lines.append(f"Total: {total} records")
    return {"collections": rows, "total": total}, "\n".join(lines)


def _target_host(settings: WeaviateSettings) -> str:
    """The normalized host and port of `settings.http`.

    `connect` uses this host. Without WEAVIATE_GRPC_URL the client always uses
    port 443, so an explicit port in WEAVIATE_URL is stored but not used.
    """
    return settings.http.normalized


def _state_mismatch(directory: Path, copy: str, now: str) -> CliError:
    return CliError(
        "state_mismatch",
        f"The state in {directory} holds a copy of {copy}. {now}",
        STATE_HINT,
    )


def _planned_targets(
    state: State,
    reader: TargetReader,
    entries: list[CollectionPlan],
    host: str,
    source_names: dict[str, str],
    directory: Path,
) -> dict[str, Target]:
    """The target of each entry by source ref, after a check against the state.

    Raises state_mismatch when the entry has progress and the state holds its
    copy to another target, when the state holds this target for another
    source, or when the entry has progress and its collection or tenant does
    not exist in the target. An entry without progress is a first run: its
    planned target replaces the stored one.
    """
    stored = state.targets()
    planned: dict[str, Target] = {}
    for entry in entries:
        ref, name = entry.source.ref, entry.source.name
        target = Target(host, collection_key(entry.target.name), entry.target.tenant)
        progress = state.has_progress(ref)
        previous = stored.get(ref)
        if progress and previous is not None and previous != target:
            raise _state_mismatch(
                directory,
                f"{name} to {_target_text(previous)}",
                f"The plan and WEAVIATE_URL now name {_target_text(target)}.",
            )
        other = next((r for r, t in stored.items() if r != ref and t == target), None)
        if other is not None:
            raise _state_mismatch(
                directory,
                f"{source_names.get(other, other)} to {_target_text(target)}",
                f"The plan now copies {name} there.",
            )
        if progress and not reader.exists(entry):
            missing = "collection or tenant" if target.tenant else "collection"
            raise _state_mismatch(
                directory,
                f"{name} to {_target_text(target)}",
                f"That {missing} does not exist in the target now.",
            )
        planned[ref] = target
    return planned


@contextmanager
def _copy_lock(directory: Path) -> Iterator[None]:
    """Hold the lock that lets one copy at a time use `directory`."""
    try:
        with exclusive_lock(directory / LOCK_FILE):
            yield
    except Locked as exc:
        raise CliError(
            "usage",
            f"Another to-weaviate copy uses {directory}.",
            "Wait for it to end, or use another --dir.",
        ) from exc


def _select(entries: list[CollectionPlan], names: list[str]) -> list[CollectionPlan]:
    if not names:
        return entries
    known = {n for e in entries for n in (e.source.name, e.target.name)}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise CliError(
            "usage",
            f"--collection {', '.join(unknown)} matches no plan entry.",
            "Use a source or target collection name from plan.yaml.",
        )
    return [e for e in entries if e.source.name in names or e.target.name in names]


def _copy(
    factories: _Factories,
    directory: Path,
    names: list[str],
    parallelism: int,
    consistency: Consistency | None,
    json_output: bool,
) -> None:
    if not 1 <= parallelism <= MAX_PARALLELISM:
        raise CliError(
            "usage",
            f"--parallelism {parallelism} is outside 1 to {MAX_PARALLELISM}.",
            f"Use a --parallelism from 1 to {MAX_PARALLELISM}.",
        )
    plan, inspected = _load_plan_and_inspection(directory)
    entries = _select(plan.collections, names)
    groups = target_groups(plan.collections)
    selected = {e.target.name: groups[e.target.name] for e in entries}
    keys = {collection_key(name) for name in selected}
    errors, warnings = plan_findings(plan, inspected, vectorizer_for)
    errors = [f for f in errors if collection_key(f.collection) in keys]
    if errors:
        warnings = [w for w in warnings if collection_key(w.collection) in keys]
        raise _plan_errors(errors, warnings)
    env = load_env(Path.cwd())
    source_settings = chroma_settings(env)
    target_settings = weaviate_settings(env)
    _require_provider_keys(
        plan.model_copy(update={"collections": entries}), target_settings
    )
    with _copy_lock(directory):
        source = factories.source(source_settings)
        schema = factories.schema(target_settings)
        writer = factories.writer(target_settings, consistency)
        reader = factories.reader(target_settings)
        source_names = {c.ref: c.name for c in inspected.values()}
        source_names.update({e.source.ref: e.source.name for e in plan.collections})
        state = State(directory / STATE_FILE)
        try:
            targets = _planned_targets(
                state,
                reader,
                entries,
                _target_host(target_settings),
                source_names,
                directory,
            )
            for group in selected.values():
                schema.ensure(group)
            events = Events(directory / EVENTS_FILE)
            for entry in entries:
                state.set_target(entry.source.ref, targets[entry.source.ref])
                info(json_output, f"Copying {entry.source.name} to {entry.target.name}")
                result = copy_collection(
                    source,
                    writer,
                    state,
                    events,
                    entry,
                    inspected[entry.source.ref],
                    parallelism,
                )
                info(
                    json_output,
                    f"Copied {entry.source.name}: {_plural(result.copied, 'record')}, "
                    f"{_plural(result.rejects, 'reject')}",
                )
            progress = {p.ref: p for p in state.progress()}
        finally:
            state.close()
    rows = _rows(entries, progress)
    document, text = _progress_output(entries, rows)
    rejects = sum(row["rejects"] for row in rows)
    if rejects:
        raise CliError(
            "rejects",
            f"{_plural(rejects, 'record')} rejected.",
            f"See {directory / REJECTS_FILE} for each record and the reason.",
            fields=document,
            text=text,
        )
    emit(json_output, {"status": "success", **document}, text)


def _status(directory: Path, json_output: bool) -> None:
    plan = _load_plan(directory)
    path = directory / STATE_FILE
    progress: dict[str, CollectionProgress] = {}
    targets: dict[str, Target] = {}
    if path.exists():
        state = State(path)
        try:
            progress = {p.ref: p for p in state.progress()}
            targets = state.targets()
        finally:
            state.close()
    rows = _rows(plan.collections, progress)
    document, text = _progress_output(plan.collections, rows, targets)
    emit(json_output, {"status": "success", **document}, text)


def _verify(
    factories: _Factories,
    directory: Path,
    sample: int,
    quick: bool,
    json_output: bool,
) -> None:
    # A run that fails must not leave the report of an earlier run.
    remove_report(directory)
    plan, inspected = _load_plan_and_inspection(directory)
    env = load_env(Path.cwd())
    source_settings = chroma_settings(env)
    target_settings = weaviate_settings(env)
    source = factories.source(source_settings)
    reader = factories.reader(target_settings)
    rng = random.Random()
    checks: list[Check] = []
    for entry in plan.collections:
        collection = inspected[entry.source.ref]
        place = target_label(entry.target.name, entry.target.tenant)
        info(json_output, f"Verifying {entry.source.name} in {place}")
        missing = missing_target(reader, entry)
        if missing is not None:
            checks.append(missing)
            continue
        checks.append(verify_counts(source, reader, entry, collection))
        if not quick:
            diff = diff_ids(source, reader, entry, collection, directory)
            checks.append(verify_ids(diff, entry))
        checks.append(verify_records(source, reader, entry, collection, sample, rng))
    write_report(checks, directory)
    report = directory / REPORT_MD
    document = {
        "checks": [dataclasses.asdict(c) for c in checks],
        "report": str(report),
    }
    text = "\n".join(check_line(c) for c in checks)
    failed = sum(not c.passed for c in checks)
    if failed:
        raise CliError(
            "verification_failed",
            f"{_plural(failed, 'check')} failed.",
            f"See {report} for the detail of each check.",
            fields=document,
            text=text,
        )
    emit(json_output, {"status": "success", **document}, text)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"to-weaviate {__version__}")
        raise typer.Exit()


def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    pass


DirOption = Annotated[Path, typer.Option("--dir", help="Migration directory.")]
JsonOption = Annotated[
    bool, typer.Option("--json", help="Print one JSON document on stdout.")
]
NamesOption = Annotated[
    list[str] | None,
    typer.Option("--collection", metavar="NAME", help="Collection name. Repeatable."),
]


def build_app(
    source_factory: SourceFactory,
    writer_factory: WriterFactory,
    reader_factory: ReaderFactory,
    schema_factory: SchemaFactory,
) -> typer.Typer:
    factories = _Factories(
        source_factory, writer_factory, reader_factory, schema_factory
    )
    # Locals can hold values read from the environment, so tracebacks omit them.
    app = typer.Typer(
        help="Migrate your data to Weaviate from other vector databases.",
        no_args_is_help=True,
        pretty_exceptions_show_locals=False,
    )
    app.callback()(main)
    inspect_app = typer.Typer(
        help="Inspect a source and draft plan.yaml.", no_args_is_help=True
    )
    app.add_typer(inspect_app, name="inspect")

    @inspect_app.command("chroma")
    def inspect_chroma(
        database: Annotated[
            list[str] | None,
            typer.Option(
                "--database",
                metavar="NAME",
                help="Database. Repeatable. Default: CHROMA_DATABASE, else all.",
            ),
        ] = None,
        collection: NamesOption = None,
        sample_pages: Annotated[
            int | None,
            typer.Option(
                "--sample-pages",
                min=1,
                metavar="N",
                help="Metadata pages to sample per collection. Default: all.",
            ),
        ] = None,
        vectors: Annotated[
            VectorsMode,
            typer.Option("--vectors", help="Keep the source vectors, embed, or both."),
        ] = "keep",
        vectorizer: Annotated[
            str | None,
            typer.Option("--vectorizer", metavar="MODULE", help="Weaviate module."),
        ] = None,
        vectorizer_option: Annotated[
            list[str] | None,
            typer.Option(
                "--vectorizer-option",
                metavar="KEY=VALUE",
                help="Vectorizer module option. Repeatable.",
            ),
        ] = None,
        index: Annotated[
            IndexType, typer.Option("--index", help="Vector index type.")
        ] = "default",
        quantizer: Annotated[
            QuantizerType, typer.Option("--quantizer", help="Vector quantizer.")
        ] = "default",
        quantizer_bits: Annotated[
            int | None,
            typer.Option("--quantizer-bits", metavar="1|4|8", help="Bits for rq."),
        ] = None,
        force: Annotated[
            bool, typer.Option("--force", help="Replace plan.yaml.")
        ] = False,
        directory: DirOption = DEFAULT_DIR,
        json_output: JsonOption = False,
    ) -> None:
        """Inspect Chroma collections and write inspect.json and plan.yaml."""
        with _handled(json_output):
            _inspect_chroma(
                factories,
                directory,
                database or [],
                collection or [],
                sample_pages,
                vectors,
                vectorizer,
                vectorizer_option or [],
                index,
                quantizer,
                quantizer_bits,
                force,
                json_output,
            )

    @app.command("check")
    def check(
        directory: DirOption = DEFAULT_DIR, json_output: JsonOption = False
    ) -> None:
        """Check plan.yaml against the target. Creates nothing."""
        with _handled(json_output):
            _check(factories, directory, json_output)

    @app.command("copy")
    def copy(
        collection: NamesOption = None,
        parallelism: Annotated[
            int,
            typer.Option(
                "--parallelism",
                metavar="N",
                help=f"Reads per collection, 1 to {MAX_PARALLELISM}.",
            ),
        ] = DEFAULT_PARALLELISM,
        consistency: Annotated[
            Consistency | None,
            typer.Option("--consistency", help="Write consistency level."),
        ] = None,
        directory: DirOption = DEFAULT_DIR,
        json_output: JsonOption = False,
    ) -> None:
        """Create the target collections, then copy the records. Resumable."""
        with _handled(json_output):
            _copy(
                factories,
                directory,
                collection or [],
                parallelism,
                consistency,
                json_output,
            )

    @app.command("status")
    def status(
        directory: DirOption = DEFAULT_DIR, json_output: JsonOption = False
    ) -> None:
        """Show copy progress from state.sqlite. Needs no credentials."""
        with _handled(json_output):
            _status(directory, json_output)

    @app.command("verify")
    def verify(
        sample: Annotated[
            int,
            typer.Option(
                "--sample",
                min=1,
                max=MAX_SAMPLE,
                metavar="N",
                help=f"Records to compare per collection, 1 to {MAX_SAMPLE}.",
            ),
        ] = DEFAULT_SAMPLE,
        quick: Annotated[
            bool, typer.Option("--quick", help="Skip the full id comparison.")
        ] = False,
        directory: DirOption = DEFAULT_DIR,
        json_output: JsonOption = False,
    ) -> None:
        """Compare the target with the source. Writes report.md and report.json."""
        with _handled(json_output):
            _verify(factories, directory, sample, quick, json_output)

    return app


_client: weaviate.WeaviateClient | None = None


def _weaviate_client(settings: WeaviateSettings) -> weaviate.WeaviateClient:
    """One client per process, shared by the writer, reader and schema."""
    global _client
    if _client is None:
        _client = connect(settings)
        atexit.register(_client.close)
    return _client


def _chroma_source(settings: ChromaSettings) -> Source:
    client = ChromaClient(settings)
    atexit.register(client.close)
    tenant = settings.tenant or client.identity()["tenant"]
    return ChromaSource(client, tenant)


def _weaviate_writer(settings: WeaviateSettings, consistency: str | None) -> Writer:
    return WeaviateWriter(_weaviate_client(settings), consistency)


def _weaviate_reader(settings: WeaviateSettings) -> TargetReader:
    return WeaviateReader(_weaviate_client(settings))


def _weaviate_schema(settings: WeaviateSettings) -> Schema:
    return WeaviateSchema(_weaviate_client(settings))


app = build_app(_chroma_source, _weaviate_writer, _weaviate_reader, _weaviate_schema)
