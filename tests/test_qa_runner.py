from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import qa_runner


@pytest.fixture(autouse=True)
def _patch_ensure_uv(monkeypatch):
    monkeypatch.setattr(qa_runner, "ensure_uv", lambda: None)


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_run_checks_success(monkeypatch, tmp_path):
    calls: list[tuple[str, ...]] = []

    def fake_run(args, capture_output: bool, text: bool):
        calls.append(tuple(args))
        index = len(calls)
        return _completed(0, stdout=f"ok-{index}")

    monkeypatch.setattr(qa_runner.subprocess, "run", fake_run)
    output_path = tmp_path / "reports" / "qa.json"
    buffer = io.StringIO()

    summary = qa_runner.run_checks(report_format="json", output_path=output_path, stream=buffer)

    assert summary.exit_code == 0
    assert all(result.succeeded for result in summary.results)
    assert len(calls) == len(qa_runner.COMMANDS)

    payload = json.loads(summary.report)
    assert payload["status"] == "success"
    assert payload["exit_code"] == 0
    assert [tuple(entry["command"]) for entry in payload["results"]] == calls

    assert output_path.read_text(encoding="utf-8") == summary.report
    assert buffer.getvalue() == summary.report


def test_run_checks_failure(monkeypatch):
    codes = iter([0, 1, 0, 0, 0])
    outputs = {1: "lint error"}
    seen = []

    def fake_run(args, capture_output: bool, text: bool):
        code = next(codes)
        seen.append(tuple(args))
        return _completed(code, stdout=f"stdout-{code}", stderr=outputs.get(code, ""))

    monkeypatch.setattr(qa_runner.subprocess, "run", fake_run)
    buffer = io.StringIO()

    summary = qa_runner.run_checks(report_format="stdout", stream=buffer)

    assert summary.exit_code == 1
    assert any(not result.succeeded for result in summary.results)
    assert "FAILED" in summary.report
    assert "Final status: FAILURE" in summary.report
    assert seen == [tuple(spec.args) for spec in qa_runner.COMMANDS]


def test_main_passes_arguments(monkeypatch, tmp_path):
    recorded: dict[str, object] = {}

    def fake_run_checks(*, report_format: str, output_path: Path | None, stream=None):
        recorded["format"] = report_format
        recorded["output"] = output_path
        return qa_runner.RunSummary(results=[], report="", exit_code=0)

    monkeypatch.setattr(qa_runner, "run_checks", fake_run_checks)

    target = tmp_path / "qa.md"
    exit_code = qa_runner.main(["--format", "markdown", "--output", str(target)])

    assert exit_code == 0
    assert recorded == {"format": "markdown", "output": target}
