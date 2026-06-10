#!/usr/bin/env python3
"""Run converted EFT workflow scripts from a JSON configuration."""

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
        config = json.load(handle)
    return config


def command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def resolve_config_path(config_path: Path, maybe_relative: str | None) -> Path:
    if not maybe_relative:
        return config_path
    path = Path(maybe_relative)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def run_stage(stage: dict[str, Any], workflow_config_path: Path, base_env: dict[str, str], python_bin: str, log_dir: Path) -> None:
    name = stage["name"]
    script = (WORKFLOW_DIR / stage["script"]).resolve()
    if not script.exists():
        raise FileNotFoundError(f"Stage {name!r} script does not exist: {script}")
    stage_config_path = resolve_config_path(workflow_config_path, stage.get("config"))
    stage_config = load_config(stage_config_path)

    env = os.environ.copy()
    env.update(base_env)
    env.update({str(k): str(v) for k, v in stage_config.get("env", {}).items()})
    env.update({str(k): str(v) for k, v in stage.get("env", {}).items()})
    env["EFT_WORKFLOW_CONFIG"] = str(stage_config_path)

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{name}.log"
    command = [python_bin, str(script)]

    print(f"\n=== {name} ===", flush=True)
    print(f"script: {script}", flush=True)
    print(f"config: {stage_config_path}", flush=True)
    print(f"log:    {log_path}", flush=True)
    print(f"cmd:    {command_text(command)}", flush=True)

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
        raise RuntimeError(f"Stage {name!r} failed with exit code {returncode} after {elapsed:.1f}s. See {log_path}")
    print(f"Completed {name} in {elapsed:.1f}s", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to workflow JSON config")
    parser.add_argument("--dry-run", action="store_true", help="Print enabled stages without executing them")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    configured_python = config.get("python")
    python_bin = sys.executable if configured_python in (None, "auto") else str(configured_python)
    log_dir = Path(config.get("log_dir", "logs/script_workflow_logs"))
    if not log_dir.is_absolute():
        log_dir = (PROJECT_DIR / log_dir).resolve()

    base_env = {str(k): str(v) for k, v in config.get("env", {}).items()}
    print("Workflow:", config.get("name", config_path.stem))
    print("Config:", config_path)
    print("Project:", PROJECT_DIR)
    print("Python:", python_bin)
    print("Log dir:", log_dir)
    print("Base env:", base_env)

    for stage in config.get("stages", []):
        if not stage.get("enabled", True):
            print(f"Skipping disabled stage: {stage.get('name', stage.get('script'))}")
            continue
        if args.dry_run:
            script = (WORKFLOW_DIR / stage["script"]).resolve()
            stage_config_path = resolve_config_path(config_path, stage.get("config"))
            print(f"Would run {stage['name']}: {script}")
            print(f"  config: {stage_config_path}")
            continue
        run_stage(stage, config_path, base_env, python_bin, log_dir)

    print("\nDry run complete." if args.dry_run else "\nWorkflow complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
