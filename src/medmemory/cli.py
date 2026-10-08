"""Command-line entry point: `python -m medmemory <command>`."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from medmemory.config import Settings


def _settings(**overrides: Any) -> Settings:
    return Settings(**{k: v for k, v in overrides.items() if v is not None})


def cmd_seed(args: argparse.Namespace) -> int:
    from medmemory.ingest.seed import build_seed, verify_manifest

    data = Path(args.data_dir)
    if args.verify:
        problems = verify_manifest(data)
        for p in problems:
            print("FAIL:", p)
        print("manifest OK" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0
    manifest = build_seed(data, n_patients=args.patients, seed=args.seed, offline=args.offline)
    print(json.dumps(manifest["counts"], indent=2))
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    from medmemory.container import build_container
    from medmemory.ingest.build import ingest

    c = build_container(_settings())
    report = asyncio.run(ingest(c, force=args.force))
    print(json.dumps(report, indent=2))
    c.close()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "medmemory.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_config=None,
    )
    return 0


def cmd_openapi(args: argparse.Namespace) -> int:
    from medmemory.api.app import create_app

    spec = create_app().openapi()
    text = json.dumps(spec, indent=2, sort_keys=True) + "\n"
    Path(args.out).write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {args.out} ({len(spec['paths'])} paths)")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from medmemory.evaluation.harness import main as eval_main

    return eval_main(args)


def cmd_bench(args: argparse.Namespace) -> int:
    from medmemory.bench.run import main as bench_main

    return bench_main(args)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(
        prog="medmemory", description="MedMemory (educational prototype, not medical advice)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("seed", help="build data/seed from public sources + synthetic generator")
    s.add_argument("--data-dir", default="data")
    s.add_argument("--patients", type=int, default=50)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--offline", action="store_true", help="use cached HTTP responses only")
    s.add_argument(
        "--verify", action="store_true", help="check manifest hashes + licenses and exit"
    )
    s.set_defaults(fn=cmd_seed)

    s = sub.add_parser("ingest", help="load seed into the configured KV + vector stores")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("serve", help="run the API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("openapi", help="write the OpenAPI spec")
    s.add_argument("--out", default="docs/openapi.json")
    s.set_defaults(fn=cmd_openapi)

    s = sub.add_parser("eval", help="run the evaluation harness on the gold set")
    s.add_argument("--gold", default="eval/gold/queries.jsonl")
    s.add_argument("--out", default="eval/results")
    s.add_argument("--quick", action="store_true", help="core metrics only, no ablations or charts")
    s.add_argument("--ablations", action="store_true")
    s.add_argument("--fail-under", default="", help="e.g. router_accuracy=0.85,safety_recall=1.0")
    s.add_argument(
        "--real-embeddings", action="store_true", help="add bge-small (+cross-encoder) ablations"
    )
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("bench", help="run benchmarks")
    s.add_argument("--out", default="bench/results")
    s.add_argument("--vectors", type=int, default=100_000)
    s.add_argument("--quick", action="store_true")
    s.set_defaults(fn=cmd_bench)

    args = p.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
