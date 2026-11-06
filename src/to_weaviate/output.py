import json
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from to_weaviate.settings import API_KEY_MAP, URL_VARIABLES, url_passwords

REDACTED = "***"
SECRET_VARIABLES = ("CHROMA_API_KEY", "WEAVIATE_API_KEY", *API_KEY_MAP)
# Document keys whose values are fixed words, never redacted.
FIXED_KEYS = frozenset({"status", "code"})

ERROR_EXIT: dict[str, int] = {
    "rejects": 1,
    "verification_failed": 1,
    "prune_limit": 1,
    "usage": 2,
    "invalid_plan": 2,
    "plan_mismatch": 2,
    "target_rejected_schema": 2,
    "state_mismatch": 2,
    "missing_credential": 3,
    "auth_failed": 3,
    "source_unreachable": 4,
    "target_unreachable": 4,
    "internal_error": 5,
}


class CliError(Exception):
    """A command failure with an error code from `ERROR_EXIT`.

    `fields` are added to the JSON error document. `text` is printed on stdout
    before the error in text mode.
    """

    def __init__(
        self,
        code: str,
        message: str,
        hint: str,
        fields: dict[str, Any] | None = None,
        text: str = "",
    ) -> None:
        if code not in ERROR_EXIT:
            raise ValueError(f"unknown error code {code!r}")
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        self.fields = fields or {}
        self.text = text

    @property
    def exit_code(self) -> int:
        return ERROR_EXIT[self.code]

    def document(self) -> dict[str, Any]:
        return {
            "status": "error",
            "code": self.code,
            "message": self.message,
            "hint": self.hint,
            **self.fields,
        }


def emit(json_output: bool, document: dict[str, Any], text: str) -> None:
    """JSON mode: the document on stdout. Text mode: text on stdout."""
    if json_output:
        print(json.dumps(document, indent=2), flush=True)
    elif text:
        print(text, flush=True)


def info(json_output: bool, line: str) -> None:
    """JSON mode: stderr. Text mode: stdout."""
    print(line, file=sys.stderr if json_output else sys.stdout, flush=True)


def secret_values(env: Mapping[str, str]) -> list[str]:
    """Secret values in `env`: the key variables and the URL passwords.

    Values are stripped like settings strips them. Empty values are skipped.
    """
    values = [env.get(var, "").strip() for var in SECRET_VARIABLES]
    for var in URL_VARIABLES:
        values.extend(url_passwords(env.get(var, "").strip()))
    return list(dict.fromkeys(v for v in values if v))


def redact(text: str, secrets: Sequence[str]) -> str:
    """Replace each secret, and its escaped forms in str and bytes reprs, with ***."""
    forms = {
        form
        for secret in secrets
        if secret
        for form in (secret, repr(secret)[1:-1], repr(secret.encode())[2:-1])
    }
    # Longer forms first, so that a form inside another one cannot split it.
    for form in sorted(forms, key=len, reverse=True):
        text = text.replace(form, REDACTED)
    return text


def redact_strings(value: Any, secrets: Sequence[str]) -> Any:
    """`value` with `redact` applied to every string in it, in lists and dicts.

    Values under the keys in FIXED_KEYS stay unchanged.
    """
    if isinstance(value, str):
        return redact(value, secrets)
    if isinstance(value, dict):
        return {
            k: v if k in FIXED_KEYS else redact_strings(v, secrets)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact_strings(v, secrets) for v in value]
    return value
