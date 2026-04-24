"""
Subprocess sandbox for executing model-generated Python code.
Uses a worker pool for parallel execution.
"""

import concurrent.futures
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

TIMEOUT = 120  # seconds per execution


def _run_code(code: str, timeout: int = TIMEOUT) -> dict:
    """Run Python code in an isolated subprocess. Returns {'stdout', 'stderr', 'ok'}."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "ok": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "TimeoutExpired", "ok": False}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "ok": False}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def execute_code(code: str, timeout: int = TIMEOUT) -> dict:
    """Execute a single code block. Returns {'stdout', 'stderr', 'ok', 'result'}."""
    out = _run_code(code, timeout)
    result_val = None
    if out["ok"]:
        for line in out["stdout"].splitlines():
            if line.startswith("result="):
                result_val = line[len("result="):].strip()
                break
    out["result"] = result_val
    return out


def execute_batch(codes: list[str], max_workers: int = 8, timeout: int = TIMEOUT) -> list[dict]:
    """Execute a list of code strings in parallel. Returns list of result dicts."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(execute_code, c, timeout) for c in codes]
        return [f.result() for f in futures]
