import json
import sys
from typing import Any

ERROR_EXIT: dict[str, int] = {
    "rejects": 1,
    "verification_failed": 1,
    "prune_limit": 1,
    "usage": 2,
    "invalid_plan": 2,
    "plan_mismatch": 2,
    "target_rejected_schema": 2,
    "missing_credential": 3,
    "auth_failed": 3,
    "source_unreachable": 4,
    "target_unreachable": 4,
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
