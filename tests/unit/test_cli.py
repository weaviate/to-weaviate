from typer.testing import CliRunner

from to_weaviate.cli import app


def test_version():
    r = CliRunner().invoke(app, ["--version"])
    assert r.exit_code == 0 and r.output.startswith("to-weaviate ")
