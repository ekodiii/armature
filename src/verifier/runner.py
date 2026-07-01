"""vr-test-runner — execute a synthesized TestProgram against real code.

The un-fakeable step of the verification-router MVP. It writes the generated
pytest + Hypothesis module to an isolated temp directory, runs it in a
subprocess under a timeout, and collects a structured RawTestOutcome. The run
-- not the LLM's say-so -- is the ground truth that gates the obligation
verdict.

Isolation: the subprocess runs with cwd set to the temp dir, so pytest takes
that dir as its rootdir and does NOT discover the host repo's pyproject /
testpaths / conftest. PYTHONPATH is augmented with the TestProgram's
import_roots so the generated test can import the real leaf under test.
"""

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .models import RawTestOutcome, TestProgram

# pytest exit codes we care about. The obligation verdict
# (verified / refuted / error) is vr-result-interpreter's job; the runner only
# reports what pytest mechanically did.
_PYTEST_OK = 0        # all collected tests passed
_PYTEST_FAILED = 1    # tests collected, at least one failed (e.g. a falsification)
_PYTEST_NO_TESTS = 5  # nothing collected -> the program was empty / malformed

def _sanitize(obligation_id: str) -> str:
    """A filesystem- and pytest-collection-safe stem (must match test_*.py)."""
    slug = re.sub(r"[^0-9A-Za-z_]+", "_", obligation_id).strip("_")
    return slug or "obligation"


def _strip_report_prefix(line: str) -> str:
    """Drop pytest's failure-line marker ('E' + spaces) if present, keeping the
    indentation that follows so continuation lines are still recognisable."""
    return line[1:] if line[:1] == "E" else line


def _extract_falsifying_example(stdout: str) -> str | None:
    """Pull Hypothesis's 'Falsifying example:' block out of captured stdout.

    Under pytest each line of the failure detail is prefixed with 'E   ', so we
    strip that marker and then capture from 'Falsifying example:' through the
    contiguous indented continuation lines (the args and the closing paren).
    """
    lines = stdout.splitlines()
    idx = next((i for i, ln in enumerate(lines)
                if "Falsifying example:" in ln), None)
    if idx is None:
        return None

    head = _strip_report_prefix(lines[idx])
    block = [head[head.index("Falsifying example:"):].rstrip()]
    for ln in lines[idx + 1:]:
        content = _strip_report_prefix(ln)
        if content.strip() == "" or not content[:1].isspace():
            break  # blank line or a new (unindented) report section
        block.append(content.rstrip())
        if content.strip() == ")":
            break  # closing paren of the example block
    return "\n".join(block).strip()


def _status_from_returncode(returncode: int) -> str:
    if returncode == _PYTEST_OK:
        return "passed"
    if returncode == _PYTEST_FAILED:
        return "failed"
    # 5 = nothing collected; 2/3/4 = interrupted / internal / usage error.
    # None of these is a genuine property verdict.
    return "error"


def _as_text(captured) -> str:
    if captured is None:
        return ""
    if isinstance(captured, bytes):
        return captured.decode(errors="replace")
    return captured


def run_test_program(
    program: TestProgram,
    *,
    timeout_s: float = 30.0,
    python: str | None = None,
) -> RawTestOutcome:
    """Run a TestProgram in an isolated subprocess and capture the outcome.

    - program:    the generated pytest + Hypothesis module.
    - timeout_s:  wall-clock budget; on expiry the status is 'timeout'.
    - python:     interpreter to run pytest with; defaults to the current one
                  (which must have pytest + hypothesis + the target importable).
    """
    interpreter = python or sys.executable

    env = os.environ.copy()
    roots = [r for r in program.import_roots if r]
    if roots:
        existing = env.get("PYTHONPATH", "")
        parts = [*roots, existing] if existing else list(roots)
        env["PYTHONPATH"] = os.pathsep.join(parts)

    with tempfile.TemporaryDirectory(prefix="vr_pbt_") as tmp:
        test_path = Path(tmp) / f"test_{_sanitize(program.obligation_id)}.py"
        test_path.write_text(program.source)
        cmd = [
            interpreter, "-m", "pytest",
            "-q", "-p", "no:cacheprovider",
            str(test_path),
        ]
        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd, cwd=tmp, env=env,
                capture_output=True, text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as e:
            return RawTestOutcome(
                status="timeout",
                returncode=None,
                stdout=_as_text(e.stdout),
                stderr=_as_text(e.stderr),
                duration_s=time.monotonic() - start,
            )
        duration = time.monotonic() - start

    return RawTestOutcome(
        status=_status_from_returncode(proc.returncode),
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        falsifying_example=_extract_falsifying_example(proc.stdout),
        duration_s=duration,
    )
