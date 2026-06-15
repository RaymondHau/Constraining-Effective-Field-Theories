#!/usr/bin/env python3
"""Run one EFT workflow stage from one stage-specific JSON config."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


WORKFLOW_DIR = Path(__file__).resolve().parent
PROJECT_DIR = WORKFLOW_DIR


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Stage name used for logging")
    parser.add_argument("--script", required=True, type=Path, help="Stage Python script, relative to eft_script_workflow")
    parser.add_argument("--config", required=True, type=Path, help="Stage JSON config")
    parser.add_argument("--dry-run", action="store_true", help="Print what would run without executing")
    args = parser.parse_args()

    script = args.script if args.script.is_absolute() else (WORKFLOW_DIR / args.script).resolve()
    config_path = args.config if args.config.is_absolute() else (WORKFLOW_DIR / args.config).resolve()
    config = load_config(config_path)

    python_bin = os.environ.get("PYTHON_BIN", sys.executable)
    log_dir = Path(os.environ.get("EFT_LOG_DIR", config.get("log_dir", "logs/script_workflow_logs")))
    if not log_dir.is_absolute():
        log_dir = (PROJECT_DIR / log_dir).resolve()

    command = [python_bin, "-u", str(script)]
    print(f"Stage:  {args.name}")
    print(f"Script: {script}")
    print(f"Config: {config_path}")
    print(f"Python: {python_bin}")
    print(f"Logs:   {log_dir}")
    print(f"Cmd:    {command_text(command)}")

    if args.dry_run:
        return 0

    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in config.get("env", {}).items()})
    env["EFT_WORKFLOW_CONFIG"] = str(config_path)
    env["PYTHONUNBUFFERED"] = "1"

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{args.name}.log"
    print(f"Log:    {log_path}", flush=True)

    start = time.monotonic()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"$ {command_text(command)}\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=PROJECT_DIR,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        returncode = process.wait()

    elapsed = time.monotonic() - start
    if returncode != 0:
        raise RuntimeError(f"Stage {args.name!r} failed with exit code {returncode} after {elapsed:.1f}s. See {log_path}")
    print(f"Completed {args.name} in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
