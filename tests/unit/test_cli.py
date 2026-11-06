import subprocess
import sys
import sysconfig
from pathlib import Path

from typer.testing import CliRunner

from to_weaviate.cli import app


def test_version():
    r = CliRunner().invoke(app, ["--version"])
    assert r.exit_code == 0 and r.output.startswith("to-weaviate ")


def test_installed_command_prints_no_warning():
    command = Path(sysconfig.get_path("scripts")) / "to-weaviate"
    r = subprocess.run(
        [str(command), "--version"], capture_output=True, text=True, check=False
    )
    assert r.returncode == 0 and r.stdout.startswith("to-weaviate ")
    assert r.stderr == ""


# weaviate imports `deprecate` from authlib.deprecate, so setting the module to
# None would stop weaviate from importing. The stub keeps `deprecate` and has no
# AuthlibDeprecationWarning, so only the CLI's import of that name fails.
WITHOUT_AUTHLIB_WARNING = """
import sys, types
stub = types.ModuleType("authlib.deprecate")
stub.deprecate = lambda *args, **kwargs: None
sys.modules["authlib.deprecate"] = stub
from to_weaviate.cli import app
app()
"""


def test_cli_starts_when_authlib_has_no_warning_class():
    r = subprocess.run(
        [sys.executable, "-c", WITHOUT_AUTHLIB_WARNING, "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("to-weaviate ")
