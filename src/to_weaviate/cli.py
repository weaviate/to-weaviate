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
import errno
import json
import os
import random
import re
import sqlite3
import sys
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
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
from to_weaviate.engine.copy import (
    REJECTS_HINT,
    copy_collection,
    raise_if_compacted,
)
from to_weaviate.engine.sync import (
    PRUNE_LIMIT_PERCENT,
    SourceWriting,
    SyncResult,
    sync_collection,
)
from to_weaviate.engine.verify import (
    REPORT_MD,
    Check,
    check_line,
    diff_ids,
    foreign_count,
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
    key_variables,
    load_env,
    stored_url,
    weaviate_settings,
    without_userinfo,
)
from to_weaviate.sources.base import (
    FieldStats,
    Source,
    SourceChanged,
    SourceCollection,
    SourceNotCompacted,
)
from to_weaviate.sources.chroma.client import (
    COLLECTION_GONE_HINT,
    ChromaClient,
    ChromaError,
)
from to_weaviate.sources.chroma.mapping import stored_embedding, vectorizer_for
from to_weaviate.sources.chroma.source import ChromaSource
from to_weaviate.state import (
    CollectionProgress,
    State,
    StateInUse,
    StateUnreadable,
    StateUnwritable,
    Target,
    unwritable_reason,
)
from to_weaviate.target.connect import connect
from to_weaviate.target.deleter import DeleteFailed, Deleter, WeaviateDeleter
from to_weaviate.target.reader import SOURCE_ID, TargetReader, WeaviateReader
from to_weaviate.target.schema import (
    USERINFO_TEXT,
    PlanMismatch,
    PropertyRejected,
    Schema,
    SchemaRejected,
    WeaviateSchema,
    nearest_grpc_status,
)
from to_weaviate.target.writer import (
    BatchRefused,
    ProviderRefused,
    WeaviateWriter,
    Writer,
)

SourceFactory = Callable[[ChromaSettings], Source]
WriterFactory = Callable[[WeaviateSettings, str | None], Writer]
ReaderFactory = Callable[[WeaviateSettings], TargetReader]
SchemaFactory = Callable[[WeaviateSettings], Schema]
DeleterFactory = Callable[[WeaviateSettings, str | None], Deleter]

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
CHROMA_REJECTED_HINT = (
    "Chroma refused the request and will refuse it again when the command runs "
    "again. The message holds the reason."
)
# Chroma 0.5 and 0.6 answer 403 for a right token in the wrong header.
CHROMA_TOKEN_HEADER_HINT = (
    "If the server reads the token from another header, set CHROMA_TOKEN_HEADER."
)
CHANGED_HINT = "Stop the writes to the collection, then run the command again."
NOT_COMPACTED_HINT = (
    "Stop the writes to the collection and run the command again once Chroma "
    "has compacted them."
)
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
    "Read the server's message and edit plan.yaml. A free Weaviate Cloud "
    "cluster accepts only the index type hfresh."
)
STATE_HINT = (
    "To resume, set WEAVIATE_URL and the target name back. To copy to the new "
    "target, use another --dir."
)
GONE_HINT = "To copy it again, use another --dir."
SOURCE_CHANGED_HINT = (
    "Run to-weaviate inspect chroma --force to inspect the new source, or set "
    "CHROMA_URL and CHROMA_TENANT back to the source of the plan."
)
UNREADABLE_STATE_HINT = "Start again with another --dir, or restore the file."
UNWRITABLE_HINT = "Make --dir and its files writable, or use another --dir."
DISK_FULL_HINT = "Free space on the disk of --dir, or use another --dir."
# The errors of a write that the disk or the permissions of --dir refuse.
UNWRITABLE_ERRNOS = {
    errno.ENOSPC: DISK_FULL_HINT,
    errno.EDQUOT: DISK_FULL_HINT,
    errno.EACCES: UNWRITABLE_HINT,
    errno.EPERM: UNWRITABLE_HINT,
    errno.EROFS: UNWRITABLE_HINT,
}
STATE_IN_USE_HINT = "Wait for it to end, then run the command again."
ADD_PROPERTY_HINT = (
    "Weaviate refused to add the property; read its message and edit plan.yaml."
)
BATCH_HINT = "Fix the cause and run to-weaviate copy again. It resumes at this page."
PRUNE_HINT = (
    f"More than {PRUNE_LIMIT_PERCENT}% of the objects with a source_id have no "
    "source record. Check that the records were deleted on purpose and that no "
    "other migration directory or application writes objects with a source_id "
    "to the collection. To delete them, run to-weaviate sync --prune --force."
)
SYNC_BATCH_HINT = "Fix the cause and run to-weaviate sync again."
PROVIDER_BUSY_HINT = (
    "Wait for the provider's rate limit to reset or for the provider to "
    "answer again, or lower --parallelism. Then run to-weaviate {command} again."
)
FOREIGN_HINT = (
    "Copy into another collection or tenant, or, when these objects may be "
    "deleted, run the same command with --yes."
)
MISSING_TEXT = "missing in the collection; copy and sync add it"
# A part of a text in single or double quotes; a quote after a backslash is
# part of it, as in a Python repr.
_QUOTED = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")


@dataclass(frozen=True)
class _Factories:
    source: SourceFactory
    writer: WriterFactory
    reader: ReaderFactory
    schema: SchemaFactory
    deleter: DeleterFactory


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


def _unwritable_hint(reason: str) -> str:
    return DISK_FULL_HINT if reason == "the disk is full" else UNWRITABLE_HINT


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


def _source_rejected(status: int) -> bool:
    """Whether the source answers the same request with HTTP `status` again.

    408, 429 and every 5xx but 501 can pass on a later run. A 3xx and a 200
    that is not JSON stay source_unreachable: the address is not a Chroma API.
    """
    return status == 501 or (400 <= status < 500 and status not in (408, 429))


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
            exc.hint or f"Set {exc.var} in the environment or in .env.",
        )
    if isinstance(exc, InvalidSetting):
        return CliError("usage", str(exc), exc.hint)
    if isinstance(exc, StateUnreadable):
        return CliError("invalid_state", str(exc), UNREADABLE_STATE_HINT)
    if isinstance(exc, StateInUse):
        return CliError("usage", str(exc), STATE_IN_USE_HINT)
    if isinstance(exc, StateUnwritable):
        return CliError("directory_unwritable", str(exc), _unwritable_hint(exc.reason))
    if isinstance(exc, sqlite3.OperationalError):
        reason = unwritable_reason(exc)
        if reason is not None:
            message = f"A database file cannot be written: {reason}."
            return CliError("directory_unwritable", message, _unwritable_hint(reason))
    if isinstance(exc, OSError) and exc.errno in UNWRITABLE_ERRNOS:
        where = "The migration directory"
        if exc.filename:
            path = Path(str(exc.filename))
            where = f"{path.name} in {path.parent}"
        return CliError(
            "directory_unwritable",
            f"{where} cannot be written: {os.strerror(exc.errno)}.",
            UNWRITABLE_ERRNOS[exc.errno],
        )
    if isinstance(exc, SourceChanged):
        changes = ", ".join(
            f"{key} {value} to {exc.after.get(key)}"
            for key, value in exc.before.items()
            if exc.after.get(key) != value
        )
        return CliError(
            "source_compacting",
            f"Chroma compacted the collection {exc.collection} while it was read "
            f"({changes}), so a page can have skipped records. Run the command "
            "again.",
            CHANGED_HINT,
            fields={"collection": exc.collection},
        )
    if isinstance(exc, SourceNotCompacted):
        return CliError(
            "source_compacting",
            f"Chroma has not compacted every write to the collection "
            f"{exc.collection} after {exc.seconds:.0f} s: {exc.unindexed} of "
            f"{exc.total} operations are not compacted. Run the command again.",
            NOT_COMPACTED_HINT,
            fields={"collection": exc.collection},
        )
    if isinstance(exc, ChromaError):
        if exc.status in (401, 403):
            refused = _refused("Chroma", "CHROMA_API_KEY", exc.status == 403)
            if exc.hint is not None:
                refused.hint = exc.hint
            if exc.status == 403:
                refused.hint = f"{refused.hint} {CHROMA_TOKEN_HEADER_HINT}"
            return refused
        if exc.status is None:
            return CliError(
                "source_unreachable", f"Chroma could not be reached: {exc}", CHROMA_HINT
            )
        # A collection created again under its name is reported with HTTP 200.
        if _source_rejected(exc.status) or exc.hint == COLLECTION_GONE_HINT:
            return CliError(
                "source_rejected", str(exc), exc.hint or CHROMA_REJECTED_HINT
            )
        return CliError("source_unreachable", str(exc), exc.hint or CHROMA_HINT)
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
    status = nearest_grpc_status(exc) if isinstance(exc, WeaviateBaseError) else None
    if status in ("UNAUTHENTICATED", "PERMISSION_DENIED"):
        return _refused("Weaviate", "WEAVIATE_API_KEY", status == "PERMISSION_DENIED")
    if isinstance(exc, ProviderRefused):
        variables = key_variables(exc.provider)
        key = " or ".join(variables) if variables else f"the key of {exc.provider}"
        return _refused(exc.provider, key, exc.status == 403)
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
            f"Weaviate refused to {exc.action}: {exc}",
            ADD_PROPERTY_HINT if isinstance(exc, PropertyRejected) else SCHEMA_HINT,
        )
    if isinstance(exc, BatchRefused):
        hint = BATCH_HINT
        if exc.provider_busy:
            hint = (
                PROVIDER_BUSY_HINT.format(command="copy") + " It resumes at this page."
            )
        return CliError("target_unreachable", str(exc), hint)
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

# The secret values of the running command; see `_command_secrets`.
_COMMAND_SECRETS: ContextVar[list[str] | None] = ContextVar(
    "command_secrets", default=None
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
    token = _COMMAND_SECRETS.set(None)
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
    finally:
        _COMMAND_SECRETS.reset(token)


def _command_secrets() -> list[str]:
    """The secret values that `_report` redacts from an error, for this command.

    The set is read on first use and kept until the command ends: `_handled`
    clears it when a command starts. That is safe because the environment and
    .env do not change while a command runs. A cache per process would not
    be: one process can run several commands with other secrets, as the tests
    do through CliRunner.
    """
    secrets = _COMMAND_SECRETS.get()
    if secrets is None:
        secrets = secret_values(_handler_env())
        _COMMAND_SECRETS.set(secrets)
    return secrets


def _emit_success(json_output: bool, document: dict[str, Any], text: str) -> None:
    """`emit`, redacting the `_command_secrets`."""
    secrets = _command_secrets()
    emit(json_output, redact_strings(document, secrets), redact(text, secrets))


def _info(json_output: bool, line: str) -> None:
    """`info`, redacting the `_command_secrets`."""
    info(json_output, redact(line, _command_secrets()))


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


def _chroma_text(url: str | None, tenant: str | None) -> str:
    where = url if url is not None else f"a URL that {USERINFO_TEXT}"
    which = f"tenant {tenant}" if tenant else "the tenant of the API key"
    return f"{where}, {which}"


def _require_planned_source(
    plan: Plan, settings: ChromaSettings, directory: Path
) -> None:
    """Refuse CHROMA_URL and CHROMA_TENANT when they name another source than the plan.

    The URLs are compared and printed in their stored form, without userinfo:
    plan.yaml can be edited by hand.
    """
    planned = (stored_url(plan.source.url), plan.source.tenant)
    current = (settings.stored_url, settings.tenant)
    if planned != current:
        raise CliError(
            "invalid_plan",
            f"{directory / PLAN_FILE} was inspected from {_chroma_text(*planned)}. "
            f"CHROMA_URL and CHROMA_TENANT now name {_chroma_text(*current)}.",
            SOURCE_CHANGED_HINT,
        )


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


def _vectorizer_plan(module: str | None, options: list[str]) -> VectorizerPlan | None:
    """The vectorizer plan of `module`.

    Each option value stays the text given. The collection is created with
    it as the type that the client declares for the option.
    """
    parsed: dict[str, Any] = {}
    for option in options:
        key, sep, value = option.partition("=")
        if not sep or not key:
            raise CliError(
                "usage",
                f"--vectorizer-option {option} is not KEY=VALUE.",
                "Write each option as --vectorizer-option KEY=VALUE.",
            )
        parsed[key] = value
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
            "A quantizer needs a named --index that supports it; "
            "--quantizer-bits takes 1, 4 or 8 with hnsw, 1 or 8 with flat and "
            "dynamic, and 1 with hfresh.",
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
    _info(json_output, f"Inspecting Chroma at {settings.stored_url}")
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
        raise _no_collection(source.tenant, source.searched_namespaces)
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
    _emit_success(json_output, document, text)


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
    _info(json_output, f"Checking the target at {without_userinfo(settings.url)}")
    schema = factories.schema(settings)
    errors.extend(module_findings(plan, schema.modules()))
    for name, group in target_groups(plan.collections).items():
        for difference in schema.differences(group) or []:
            errors.append(Finding("plan_mismatch", name, difference))
        warnings.extend(
            Finding("missing_property", name, f"property {prop}: {MISSING_TEXT}")
            for prop in schema.missing_properties(group)
        )
    if errors:
        raise _plan_errors(errors, warnings)
    lines = _findings_text(errors, warnings)
    lines.append(f"No errors, {_plural(len(warnings), 'warning')}.")
    document = {"status": "success", **_findings_document(errors, warnings)}
    _emit_success(json_output, document, "\n".join(lines))


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


def _state_mismatch(
    directory: Path, copy: str, now: str, hint: str = STATE_HINT
) -> CliError:
    return CliError(
        "state_mismatch",
        f"The state in {directory} holds a copy of {copy}. {now}",
        hint,
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
            # Setting WEAVIATE_URL or the target name back does not bring the
            # collection back, so the hint names only another --dir.
            raise _state_mismatch(
                directory,
                f"{name} to {_target_text(target)}",
                f"That {missing} does not exist in the target now.",
                GONE_HINT,
            )
        planned[ref] = target
    return planned


def _needs_quantizer_update(entry: CollectionPlan) -> bool:
    """Whether copy enables a quantizer of `entry` again after its pages.

    Without ASYNC_INDEXING, Weaviate trains such a quantizer only then.
    """
    return any(v.index.quantizer.needs_training() for v in entry.target.vectors)


@contextmanager
def _copy_lock(directory: Path) -> Iterator[None]:
    """Hold the lock that lets one copy at a time use `directory`."""
    try:
        with exclusive_lock(directory / LOCK_FILE):
            yield
    except Locked as exc:
        raise CliError(
            "usage",
            f"Another to-weaviate copy or sync uses {directory}.",
            "Wait for it to end, or use another --dir.",
        ) from exc


def _warning(name: str) -> Callable[[str], None]:
    """A function that prints a warning about the collection `name` on stderr.

    The printed line is redacted of the `_command_secrets`.
    """

    def warn(line: str) -> None:
        text = redact(f"Warning: {name}: {line}", _command_secrets())
        print(text, file=sys.stderr, flush=True)

    return warn


def _waiting(json_output: bool, name: str) -> Callable[[int, int], None]:
    """A function that prints the wait for Chroma's compaction of `name`."""

    def waiting(unindexed: int, total: int) -> None:
        _info(
            json_output,
            f"Waiting for Chroma to compact {name}: {unindexed} of {total} "
            "operations not compacted",
        )

    return waiting


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


@dataclass(frozen=True)
class _Run:
    """What a copy or a sync works on, once the plan and the target are checked."""

    inspected: dict[str, SourceCollection]
    entries: list[CollectionPlan]
    warnings: list[Finding]
    target_settings: WeaviateSettings
    source: Source
    writer: Writer
    reader: TargetReader
    schema: Schema
    state: State
    events: Events
    targets: dict[str, Target]
    added: list[dict[str, str]]


@contextmanager
def _writing(
    factories: _Factories,
    directory: Path,
    names: list[str],
    parallelism: int,
    consistency: Consistency | None,
    json_output: bool,
    command: str,
    yes: bool,
) -> Iterator[_Run]:
    """The checks before a copy or a sync, then the state under `copy.lock`.

    Raises the plan, source, provider-key, module and state errors, and
    creates the target collections. The state is closed on exit. When a first
    run meets objects that `sync --prune` would delete, it goes on only with
    `yes` or a yes at the prompt of a terminal; else it raises
    `foreign_objects` before it creates or writes anything.
    """
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
    warnings = [w for w in warnings if collection_key(w.collection) in keys]
    if errors:
        raise _plan_errors(errors, warnings)
    env = load_env(Path.cwd())
    source_settings = chroma_settings(env)
    _require_planned_source(plan, source_settings, directory)
    target_settings = weaviate_settings(env)
    _require_provider_keys(
        plan.model_copy(update={"collections": entries}), target_settings
    )
    with _copy_lock(directory):
        source = factories.source(source_settings)
        schema = factories.schema(target_settings)
        module_errors = [
            f
            for f in module_findings(plan, schema.modules())
            if collection_key(f.collection) in keys
        ]
        if module_errors:
            raise _plan_errors(module_errors, warnings)
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
            missing = {
                name: schema.missing_properties(group)
                for name, group in selected.items()
            }
            met: list[Finding] = []
            for entry in entries:
                if SOURCE_ID in missing[entry.target.name]:
                    # No object holds it yet, and Weaviate refuses a query
                    # for a property the collection does not have.
                    continue
                foreign = _foreign_objects(
                    source, reader, state, entry, inspected, directory
                )
                if foreign is not None:
                    _warning(entry.source.name)(foreign.message)
                    met.append(foreign)
            warnings.extend(met)
            if met and not yes and not _confirmed(command):
                shared = list(dict.fromkeys(f.collection for f in met))
                place = ", ".join(shared)
                verb = "holds" if len(shared) == 1 else "hold"
                raise CliError(
                    "foreign_objects",
                    f"{place} {verb} objects that sync --prune would delete; "
                    f"the {command} wrote nothing.",
                    FOREIGN_HINT,
                    fields=_findings_document([], warnings),
                )
            added: list[dict[str, str]] = []
            for name, group in selected.items():
                schema.ensure(group)
                for prop in missing[name]:
                    _info(json_output, f"Added the property {prop} to {name}")
                    added.append({"collection": name, "property": prop})
            yield _Run(
                inspected,
                entries,
                warnings,
                target_settings,
                source,
                writer,
                reader,
                schema,
                state,
                Events(directory / EVENTS_FILE),
                targets,
                added,
            )
        finally:
            state.close()


def _interactive() -> bool:
    return sys.stdin.isatty()


def _confirmed(command: str) -> bool:
    """Whether the user says yes at a prompt; False without a terminal."""
    if not _interactive():
        return False
    try:
        return typer.confirm(f"Go on with the {command}?", default=False, err=True)
    except typer.Abort:
        return False


def _foreign_objects(
    source: Source,
    reader: TargetReader,
    state: State,
    entry: CollectionPlan,
    inspected: dict[str, SourceCollection],
    directory: Path,
) -> Finding | None:
    """A warning when the first run of `entry` meets objects `--prune` would delete.

    Only a first run into an existing collection or tenant lists the ids: the
    objects of later runs are this directory's own. A tenant that is not
    active is skipped: Weaviate reads none of its objects, and the first
    write reports the tenant.
    """
    ref = entry.source.ref
    if state.has_progress(ref) or not reader.exists(entry) or not reader.active(entry):
        return None
    count = foreign_count(source, reader, entry, inspected[ref], directory)
    if not count:
        return None
    name = entry.source.name
    verb = "has" if count == 1 else "have"
    return Finding(
        "foreign_objects",
        entry.target.name,
        f"{_plural(count, 'object')} {verb} a source_id that {name} does not hold; "
        "sync --prune deletes them. Unless their records were deleted from "
        f"{name}, copy into another collection or tenant.",
    )


def _begin_entry(run: _Run, entry: CollectionPlan) -> bool:
    """Record the target of `entry`; True when its quantizers need an update."""
    ref = entry.source.ref
    run.state.set_target(ref, run.targets[ref])
    trained = _needs_quantizer_update(entry)
    if trained:
        run.state.expect_quantizer_update(ref)
    else:
        # A row from a run with an earlier plan would keep the entry not done.
        run.state.quantizers_updated(ref)
    return trained


def _enable_quantizers(run: _Run, entry: CollectionPlan, json_output: bool) -> None:
    _info(json_output, f"Enabling the trained quantizers of {entry.target.name}")
    run.schema.enable_trained_quantizers(entry)
    run.state.quantizers_updated(entry.source.ref)


def _rejects_error(
    rejects: int, directory: Path, document: dict[str, Any], text: str
) -> CliError:
    return CliError(
        "rejects",
        f"{_plural(rejects, 'record')} rejected.",
        f"See {directory / REJECTS_FILE} for each record and the reason. "
        f"{REJECTS_HINT}",
        fields=document,
        text=text,
    )


def _copy(
    factories: _Factories,
    directory: Path,
    names: list[str],
    parallelism: int,
    consistency: Consistency | None,
    json_output: bool,
    yes: bool = False,
) -> None:
    with _writing(
        factories, directory, names, parallelism, consistency, json_output, "copy", yes
    ) as run:
        rereads: dict[str, int] = {}
        for entry in run.entries:
            ref = entry.source.ref
            trained = _begin_entry(run, entry)
            _info(json_output, f"Copying {entry.source.name} to {entry.target.name}")
            result = copy_collection(
                run.source,
                run.writer,
                run.state,
                run.events,
                entry,
                run.inspected[ref],
                parallelism,
                warn=_warning(entry.source.name),
                waiting=_waiting(json_output, entry.source.name),
            )
            if result.reread_from is not None:
                rereads[ref] = result.reread_from
            if trained:
                _enable_quantizers(run, entry, json_output)
            _info(
                json_output,
                f"Copied {entry.source.name}: {_plural(result.copied, 'record')}, "
                f"{_plural(result.rejects, 'reject')}",
            )
        progress = {p.ref: p for p in run.state.progress()}
    rows = _rows(run.entries, progress)
    for row in rows:
        if row["ref"] in rereads:
            row["reread_from"] = rereads[row["ref"]]
    document, text = _progress_output(run.entries, rows)
    document["added_properties"] = run.added
    document["warnings"] = _findings_document([], run.warnings)["warnings"]
    text = "\n".join([*_findings_text([], run.warnings), text])
    rejects = sum(row["rejects"] for row in rows)
    if rejects:
        raise _rejects_error(rejects, directory, document, text)
    _emit_success(json_output, {"status": "success", **document}, text)


def _sync_output(
    entries: list[CollectionPlan], results: dict[str, SyncResult], prune: bool
) -> tuple[dict[str, Any], str]:
    rows = []
    lines = []
    for entry in entries:
        result = results[entry.source.ref]
        place = target_label(entry.target.name, entry.target.tenant)
        rows.append(
            {
                "ref": entry.source.ref,
                "source": entry.source.name,
                "target": entry.target.name,
                "tenant": entry.target.tenant,
                **{
                    key: getattr(result, key)
                    for key in (
                        "copied",
                        "unchanged",
                        "rejects",
                        "stale",
                        "deleted",
                        "plan_changed",
                        "prune_refused",
                    )
                },
            }
        )
        refused = ", prune refused" if result.prune_refused else ""
        lines.append(
            f"{entry.source.name} -> {place}: copied {result.copied}, unchanged "
            f"{result.unchanged}, rejects {result.rejects}, stale {result.stale}, "
            f"deleted {result.deleted}{refused}"
        )
    totals = {
        key: sum(getattr(r, key) for r in results.values())
        for key in ("copied", "unchanged", "rejects", "stale", "deleted")
    }
    lines.append(
        f"Total: copied {totals['copied']}, unchanged {totals['unchanged']}, "
        f"rejects {totals['rejects']}, stale {totals['stale']}, "
        f"deleted {totals['deleted']}"
    )
    if not prune and totals["stale"]:
        lines.append(
            "Run to-weaviate sync --prune to delete the "
            f"{_plural(totals['stale'], 'stale object')}."
        )
    return {"prune": prune, "collections": rows}, "\n".join(lines)


def _sync(
    factories: _Factories,
    directory: Path,
    names: list[str],
    parallelism: int,
    consistency: Consistency | None,
    prune: bool,
    force: bool,
    json_output: bool,
    yes: bool = False,
) -> None:
    if force and not prune:
        raise CliError(
            "usage", "--force needs --prune.", "Run to-weaviate sync --prune --force."
        )
    with _writing(
        factories, directory, names, parallelism, consistency, json_output, "sync", yes
    ) as run:
        deleter = factories.deleter(run.target_settings, consistency)
        results: dict[str, SyncResult] = {}
        for entry in run.entries:
            ref = entry.source.ref
            trained = _begin_entry(run, entry)
            place = target_label(entry.target.name, entry.target.tenant)
            _info(json_output, f"Syncing {entry.source.name} to {place}")
            try:
                result = sync_collection(
                    run.source,
                    run.writer,
                    run.reader,
                    deleter,
                    run.state,
                    run.events,
                    entry,
                    run.inspected[ref],
                    parallelism,
                    prune,
                    force,
                    directory,
                    warn=_warning(entry.source.name),
                    waiting=_waiting(json_output, entry.source.name),
                )
            except (SourceWriting, DeleteFailed, BatchRefused) as exc:
                raise _sync_error(exc, entry, force) from exc
            if trained:
                _enable_quantizers(run, entry, json_output)
            _info(
                json_output,
                f"Synced {entry.source.name} to {place}: "
                f"{_plural(result.copied, 'record')} written, {result.unchanged} "
                f"unchanged, {_plural(result.rejects, 'reject')}, {result.stale} "
                f"stale, {result.deleted} deleted",
            )
            results[ref] = result
    document, text = _sync_output(run.entries, results, prune)
    document["added_properties"] = run.added
    document["warnings"] = _findings_document([], run.warnings)["warnings"]
    text = "\n".join([*_findings_text([], run.warnings), text])
    refused = [
        target_label(e.target.name, e.target.tenant)
        for e in run.entries
        if results[e.source.ref].prune_refused
    ]
    limit = (
        f"Stale objects above {PRUNE_LIMIT_PERCENT}% in {', '.join(refused)}; "
        "nothing was deleted there."
    )
    rejects = sum(r.rejects for r in results.values())
    if rejects:
        error = _rejects_error(rejects, directory, document, text)
        if refused:
            error.message = f"{error.message} {limit}"
            error.hint = f"{error.hint} {PRUNE_HINT}"
        raise error
    if refused:
        raise CliError("prune_limit", limit, PRUNE_HINT, fields=document, text=text)
    _emit_success(json_output, {"status": "success", **document}, text)


def _sync_error(
    exc: SourceWriting | DeleteFailed | BatchRefused, entry: CollectionPlan, force: bool
) -> CliError:
    command = (
        "to-weaviate sync --prune --force" if force else "to-weaviate sync --prune"
    )
    if isinstance(exc, SourceWriting):
        return CliError(
            "source_writing",
            f"{exc.collection} changed while sync listed its ids: "
            f"{_plural(exc.before, 'record')} before, {exc.listed} listed, "
            f"{exc.after} after.",
            f"Stop the writes to the source, then run {command} again.",
            fields={"collection": exc.collection},
        )
    if isinstance(exc, DeleteFailed):
        place = target_label(entry.target.name, entry.target.tenant)
        return CliError(
            "delete_failed",
            f"Weaviate did not delete {exc.failed} of "
            f"{_plural(exc.matched, 'object')} of {entry.source.name} in {place}.",
            f"Run {command} again.",
            fields={"collection": entry.source.name},
        )
    hint = SYNC_BATCH_HINT
    if exc.provider_busy:
        hint = PROVIDER_BUSY_HINT.format(command="sync")
    return CliError("target_unreachable", str(exc), hint)


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
    _emit_success(json_output, {"status": "success", **document}, text)


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
    _require_planned_source(plan, source_settings, directory)
    target_settings = weaviate_settings(env)
    source = factories.source(source_settings)
    reader = factories.reader(target_settings)
    rng = random.Random()
    checks: list[Check] = []
    for entry in plan.collections:
        collection = inspected[entry.source.ref]
        place = target_label(entry.target.name, entry.target.tenant)
        _info(json_output, f"Verifying {entry.source.name} in {place}")
        missing = missing_target(reader, entry)
        if missing is not None:
            checks.append(missing)
            continue
        before = source.wait_compacted(
            collection, _waiting(json_output, entry.source.name)
        )
        checks.append(verify_counts(source, reader, entry, collection))
        if not quick:
            diff = diff_ids(source, reader, entry, collection, directory)
            checks.append(verify_ids(diff, entry))
        checks.append(verify_records(source, reader, entry, collection, sample, rng))
        raise_if_compacted(source, collection, before)
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
    _emit_success(json_output, {"status": "success", **document}, text)


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
ParallelismOption = Annotated[
    int,
    typer.Option(
        "--parallelism",
        metavar="N",
        help=f"Reads per collection, 1 to {MAX_PARALLELISM}.",
    ),
]
ConsistencyOption = Annotated[
    Consistency | None,
    typer.Option("--consistency", help="Write consistency level."),
]
YesOption = Annotated[
    bool,
    typer.Option(
        "--yes",
        help="Go on without asking when the target holds objects that "
        "sync --prune would delete.",
    ),
]
SyncConsistencyOption = Annotated[
    Consistency | None,
    typer.Option(
        "--consistency", help="Consistency level of the writes and the deletions."
    ),
]


def build_app(
    source_factory: SourceFactory,
    writer_factory: WriterFactory,
    reader_factory: ReaderFactory,
    schema_factory: SchemaFactory,
    deleter_factory: DeleterFactory,
) -> typer.Typer:
    factories = _Factories(
        source_factory, writer_factory, reader_factory, schema_factory, deleter_factory
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
        parallelism: ParallelismOption = DEFAULT_PARALLELISM,
        consistency: ConsistencyOption = None,
        yes: YesOption = False,
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
                yes,
            )

    @app.command("sync")
    def sync(
        collection: NamesOption = None,
        parallelism: ParallelismOption = DEFAULT_PARALLELISM,
        consistency: SyncConsistencyOption = None,
        prune: Annotated[
            bool,
            typer.Option(
                "--prune", help="Delete the objects whose source record is gone."
            ),
        ] = False,
        force: Annotated[
            bool,
            typer.Option(
                "--force",
                help=f"With --prune, prune a collection with more than "
                f"{PRUNE_LIMIT_PERCENT}% stale objects.",
            ),
        ] = False,
        yes: YesOption = False,
        directory: DirOption = DEFAULT_DIR,
        json_output: JsonOption = False,
    ) -> None:
        """Read the source again and write the new and changed records.

        --prune deletes the objects whose source record is gone.
        """
        with _handled(json_output):
            _sync(
                factories,
                directory,
                collection or [],
                parallelism,
                consistency,
                prune,
                force,
                json_output,
                yes,
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
    """One client per process, shared by the writer, reader, schema and deleter."""
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


def _weaviate_deleter(settings: WeaviateSettings, consistency: str | None) -> Deleter:
    return WeaviateDeleter(_weaviate_client(settings), consistency)


app = build_app(
    _chroma_source,
    _weaviate_writer,
    _weaviate_reader,
    _weaviate_schema,
    _weaviate_deleter,
)
