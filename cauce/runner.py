"""Headless, restartable submission of CAUCE window workflows to ComfyUI."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen
import uuid

from .artifacts import read_json, write_json_atomic


PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_.-]+)\s*\}\}")


def lookup(context: dict[str, Any], path: str) -> Any:
    value: Any = context
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"unknown workflow placeholder: {path}")
        value = value[part]
    return value


def materialize(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: materialize(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [materialize(item, context) for item in value]
    if not isinstance(value, str):
        return value
    exact = PLACEHOLDER.fullmatch(value)
    if exact:
        return copy.deepcopy(lookup(context, exact.group(1)))
    return PLACEHOLDER.sub(lambda match: str(lookup(context, match.group(1))), value)


class ComfyClient:
    def __init__(self, base_url: str, *, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout = float(timeout)
        self.client_id = f"cauce-{uuid.uuid4()}"
        self.headers = {"Content-Type": "application/json"}
        cf_id = os.environ.get("CAUCE_CF_ACCESS_CLIENT_ID")
        cf_secret = os.environ.get("CAUCE_CF_ACCESS_CLIENT_SECRET")
        if cf_id and cf_secret:
            self.headers["CF-Access-Client-Id"] = cf_id
            self.headers["CF-Access-Client-Secret"] = cf_secret

    def _json(self, method: str, path: str, payload: Any = None) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            urljoin(self.base_url, path.lstrip("/")),
            data=data,
            method=method,
            headers=self.headers,
        )
        with urlopen(request, timeout=self.timeout) as response:
            raw = response.read()
        return json.loads(raw.decode("utf-8")) if raw else {}

    def submit(self, workflow: dict[str, Any]) -> str:
        response = self._json(
            "POST", "/prompt", {"prompt": workflow, "client_id": self.client_id}
        )
        if response.get("error"):
            raise RuntimeError(json.dumps(response, ensure_ascii=False))
        prompt_id = response.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI returned no prompt_id: {response}")
        return str(prompt_id)

    def history(self, prompt_id: str) -> dict[str, Any] | None:
        try:
            response = self._json("GET", f"/history/{quote(prompt_id)}")
        except HTTPError as exc:
            if exc.code == 404:
                return None
            raise
        return response.get(prompt_id) if isinstance(response, dict) else None

    def wait(self, prompt_id: str, poll_seconds: float = 2.0) -> dict[str, Any]:
        while True:
            entry = self.history(prompt_id)
            if entry is not None:
                status = entry.get("status", {})
                if status.get("completed") or status.get("status_str") in {"success", "error"}:
                    if status.get("status_str") == "error":
                        raise RuntimeError(json.dumps(entry, ensure_ascii=False))
                    return entry
            time.sleep(float(poll_seconds))


def project_paths(project_path: str | Path, project: dict[str, Any]):
    source = Path(project_path).expanduser().resolve()
    root = source.parent
    workflow_path = (root / project["workflow_template"]).resolve()
    state_path = (root / project.get("state_path", ".cauce/state.json")).resolve()
    receipts_dir = (root / project.get("receipts_dir", ".cauce/runs")).resolve()
    return root, workflow_path, state_path, receipts_dir


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema": "cauce.runner-state/1", "windows": {}}
    value = read_json(path)
    if value.get("schema") != "cauce.runner-state/1":
        raise ValueError("unsupported CAUCE runner state schema")
    return value


def run_project(
    project_path: str | Path,
    *,
    resume: bool = True,
    once: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    project = read_json(project_path)
    if project.get("schema") != "cauce.project/1":
        raise ValueError("project schema must be cauce.project/1")
    root, workflow_path, state_path, receipts_dir = project_paths(project_path, project)
    workflow_template = read_json(workflow_path)
    state = load_state(state_path)
    client = None if dry_run else ComfyClient(project.get("server_url", "http://127.0.0.1:8188"))
    receipts_dir.mkdir(parents=True, exist_ok=True)

    for window in project.get("windows", []):
        window_id = str(window["id"])
        current = state["windows"].get(window_id, {})
        if resume and current.get("status") == "complete":
            continue
        context = {"project": project, "window": window}
        workflow = materialize(workflow_template, context)
        workflow_snapshot = receipts_dir / f"{window_id}.workflow.json"
        write_json_atomic(workflow_snapshot, workflow)
        if dry_run:
            state["windows"][window_id] = {
                "status": "materialized",
                "workflow": str(workflow_snapshot.relative_to(root)),
            }
            write_json_atomic(state_path, state)
            if once:
                break
            continue

        assert client is not None
        state["windows"][window_id] = {
            "status": "submitting",
            "workflow": str(workflow_snapshot.relative_to(root)),
        }
        write_json_atomic(state_path, state)
        prompt_id = client.submit(workflow)
        state["windows"][window_id].update(
            {"status": "running", "prompt_id": prompt_id, "started_at": time.time()}
        )
        write_json_atomic(state_path, state)
        try:
            history = client.wait(prompt_id, float(project.get("poll_seconds", 2.0)))
        except Exception as exc:
            state["windows"][window_id].update(
                {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            )
            write_json_atomic(state_path, state)
            raise
        receipt_path = receipts_dir / f"{window_id}.history.json"
        write_json_atomic(receipt_path, history)
        state["windows"][window_id].update(
            {
                "status": "complete",
                "finished_at": time.time(),
                "history": str(receipt_path.relative_to(root)),
            }
        )
        write_json_atomic(state_path, state)
        if once:
            break
    return state


def status_project(project_path: str | Path) -> dict[str, Any]:
    project = read_json(project_path)
    _, _, state_path, _ = project_paths(project_path, project)
    return load_state(state_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cauce", description="CAUCE ComfyUI sequence runner")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "resume"):
        command = sub.add_parser(name)
        command.add_argument("project")
        command.add_argument("--once", action="store_true")
        command.add_argument("--dry-run", action="store_true")
    status = sub.add_parser("status")
    status.add_argument("project")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "status":
            result = status_project(args.project)
        else:
            result = run_project(
                args.project,
                resume=args.command == "resume",
                once=args.once,
                dry_run=args.dry_run,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, HTTPError, URLError, RuntimeError) as exc:
        print(f"CAUCE: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
