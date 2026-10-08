"""Cross-platform task runner (Windows has no `make`). Mirrors the Makefile targets.

python tasks.py <target> [<target> ...]
python tasks.py --list
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
TASKS: dict[str, tuple[str, Callable[[], None]]] = {}


def task(help_text: str) -> Callable[[Callable[[], None]], Callable[[], None]]:
    def deco(fn: Callable[[], None]) -> Callable[[], None]:
        TASKS[fn.__name__.replace("_", "-")] = (help_text, fn)
        return fn

    return deco


def run(*cmd: str, cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    merged = {**os.environ, **(env or {})}
    rc = subprocess.call(
        list(cmd), cwd=cwd, env=merged, shell=(os.name == "nt" and cmd[0] in ("npm", "npx"))
    )
    if rc != 0:
        sys.exit(rc)


@task("Install backend (editable) with dev + eval extras")
def setup() -> None:
    run(PY, "-m", "pip", "install", "-e", ".[dev,eval]")


@task("Fetch public sources (cached) and regenerate data/seed")
def seed() -> None:
    run(PY, "-m", "medmemory", "seed")


@task("Load seed data into the configured KV + vector stores")
def ingest() -> None:
    run(PY, "-m", "medmemory", "ingest")


@task("Run the API on :8000 (mock mode unless .env says otherwise)")
def serve() -> None:
    run(PY, "-m", "medmemory", "serve", "--reload")


@task("Unit + integration tests (mock mode)")
def test() -> None:
    run(PY, "-m", "pytest", "-q", env={"MEDMEMORY_MODE": "mock"})


@task("Ruff lint + format check")
def lint() -> None:
    run(PY, "-m", "ruff", "check", "src", "tests", "tasks.py")
    run(PY, "-m", "ruff", "format", "--check", "src", "tests", "tasks.py")


@task("Auto-fix lint and formatting")
def fmt() -> None:
    run(PY, "-m", "ruff", "check", "--fix", "src", "tests", "tasks.py")
    run(PY, "-m", "ruff", "format", "src", "tests", "tasks.py")


@task("mypy type check")
def typecheck() -> None:
    run(PY, "-m", "mypy")


@task("Evaluation harness on the gold set (writes eval/results)")
def eval() -> None:
    run(PY, "-m", "medmemory", "eval", "--ablations")


@task("KV, vector and cache benchmarks (writes bench/results)")
def bench() -> None:
    run(PY, "-m", "medmemory", "bench")


@task("Regenerate docs/openapi.json from the FastAPI app")
def openapi() -> None:
    run(PY, "-m", "medmemory", "openapi", "--out", "docs/openapi.json")


@task("Remove runtime state (var/) so the next start re-ingests")
def clean() -> None:
    shutil.rmtree(ROOT / "var", ignore_errors=True)


@task("lint + typecheck + test")
def check() -> None:
    lint()
    typecheck()
    test()


def main(argv: list[str]) -> None:
    if not argv or argv[0] in ("-h", "--help", "--list"):
        width = max(map(len, TASKS))
        for name, (help_text, _) in TASKS.items():
            print(f"  {name:<{width}}  {help_text}")
        return
    for name in argv:
        if name not in TASKS:
            sys.exit(f"unknown task {name!r}; try --list")
        TASKS[name][1]()


if __name__ == "__main__":
    main(sys.argv[1:])
