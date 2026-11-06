import subprocess
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
