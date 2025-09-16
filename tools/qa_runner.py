from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, TextIO

from core.pkg_installer import ensure_uv


@dataclass(frozen=True)
class CommandSpec:
    name: str
    args: Sequence[str]


@dataclass
class CommandResult:
    name: str
    args: Sequence[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0


@dataclass
class RunSummary:
    results: list[CommandResult]
    report: str
    exit_code: int

    @property
    def status(self) -> str:
        return "success" if self.exit_code == 0 else "failure"


COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec("Install dev dependencies", ("uv", "pip", "install", "--extra", "dev", ".")),
    CommandSpec("Ruff lint", ("uv", "run", "ruff", "check", ".")),
    CommandSpec("Ruff format check", ("uv", "run", "ruff", "format", "--check", ".")),
    CommandSpec("Mypy type check", ("uv", "run", "mypy", ".")),
    CommandSpec("Pytest", ("uv", "run", "pytest", "-q")),
)


def _run_command(spec: CommandSpec) -> CommandResult:
    completed = subprocess.run(spec.args, capture_output=True, text=True)
    return CommandResult(
        name=spec.name,
        args=tuple(spec.args),
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


def _execute_commands() -> list[CommandResult]:
    ensure_uv()
    return [_run_command(spec) for spec in COMMANDS]


def _status_from_results(results: Iterable[CommandResult]) -> tuple[str, int]:
    exit_code = 0
    for result in results:
        if result.returncode != 0:
            exit_code = 1
    status = "success" if exit_code == 0 else "failure"
    return status, exit_code


def _format_command(args: Sequence[str]) -> str:
    return " ".join(args)


def _render_stdout(results: list[CommandResult]) -> str:
    status, exit_code = _status_from_results(results)
    lines = ["QA checks summary:", ""]
    for result in results:
        state = "PASSED" if result.succeeded else "FAILED"
        lines.append(
            f"- {result.name}: {state} (exit code {result.returncode})\n  Command: {_format_command(result.args)}"
        )
        if result.stdout:
            lines.append("  stdout:")
            lines.append("    " + "\n    ".join(result.stdout.rstrip().splitlines()))
        if result.stderr:
            lines.append("  stderr:")
            lines.append("    " + "\n    ".join(result.stderr.rstrip().splitlines()))
        lines.append("")
    lines.append(f"Final status: {status.upper()}")
    lines.append(f"Exit code: {exit_code}")
    return "\n".join(lines).rstrip() + "\n"


def _render_json(results: list[CommandResult]) -> str:
    status, exit_code = _status_from_results(results)
    payload = {
        "status": status,
        "exit_code": exit_code,
        "results": [
            {
                "name": result.name,
                "command": list(result.args),
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
            for result in results
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _escape_markdown(text: str) -> str:
    return text.replace("|", "\\|")


def _render_markdown(results: list[CommandResult]) -> str:
    status, exit_code = _status_from_results(results)
    lines = ["# QA Report", "", "| Step | Command | Status | Exit code |", "| --- | --- | --- | --- |"]
    for index, result in enumerate(results, start=1):
        state = "✅ Passed" if result.succeeded else "❌ Failed"
        command = _escape_markdown(_format_command(result.args))
        lines.append(f"| {index} | `{command}` | {state} | {result.returncode} |")
    lines.extend(["", f"**Final status:** {status.upper()}", f"**Exit code:** {exit_code}"])
    for index, result in enumerate(results, start=1):
        lines.extend(["", f"## Step {index}: {result.name}", "", f"**Command:** `{_escape_markdown(_format_command(result.args))}`"])
        if result.stdout:
            lines.extend(["", "<details><summary>stdout</summary>", "", "```", result.stdout.rstrip(), "```", "", "</details>"])
        if result.stderr:
            lines.extend(["", "<details><summary>stderr</summary>", "", "```", result.stderr.rstrip(), "```", "", "</details>"])
    return "\n".join(lines).rstrip() + "\n"


def render_report(results: list[CommandResult], report_format: str) -> str:
    if report_format == "stdout":
        return _render_stdout(results)
    if report_format == "json":
        return _render_json(results)
    if report_format == "markdown":
        return _render_markdown(results)
    raise ValueError(f"Unsupported report format: {report_format}")


def run_checks(
    report_format: str = "stdout",
    output_path: Path | None = None,
    stream: TextIO | None = None,
) -> RunSummary:
    if stream is None:
        stream = sys.stdout
    results = _execute_commands()
    report = render_report(results, report_format)
    _, exit_code = _status_from_results(results)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
    try:
        stream.write(report)
        if not report.endswith("\n"):
            stream.write("\n")
    except BrokenPipeError:  # pragma: no cover - stream closed
        pass
    return RunSummary(results=results, report=report, exit_code=exit_code)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="revoice-check",
        description="Run QA checks with uv helpers",
    )
    parser.add_argument(
        "--format",
        dest="report_format",
        choices=("stdout", "json", "markdown"),
        default="stdout",
        help="Output format for the report",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional file path to store the generated report",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary = run_checks(report_format=args.report_format, output_path=args.output)
    return summary.exit_code


if __name__ == "__main__":
    sys.exit(main())
