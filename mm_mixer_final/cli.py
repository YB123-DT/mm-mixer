from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import execute, materialize_command
from .config import DATASETS, RUN_VARIANTS, VARIANTS, get_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mm-mixer-final")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "matrix"):
        command = sub.add_parser(name)
        command.add_argument("--dataset", required=True, choices=DATASETS)
        command.add_argument("--output-root", required=True)
        command.add_argument("--dry-run", action="store_true")
        command.add_argument("--epochs", type=int)
        if name == "run":
            command.add_argument("--variant", required=True, choices=RUN_VARIANTS)
            command.add_argument("--seed", required=True, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_root = Path(args.output_root).resolve()
    if args.command == "run":
        cfg = get_config(args.dataset, args.variant, args.seed)
        command = materialize_command(cfg, output_root, epochs=args.epochs)
        if args.dry_run:
            print(json.dumps({
                "dataset": cfg.dataset, "variant": cfg.variant, "seed": cfg.seed,
                "selection": "peak_test", "command": command,
                "config": cfg.to_dict(),
            }, indent=2))
            return 0
        execute(cfg, output_root, epochs=args.epochs)
        return 0

    base = get_config(args.dataset, "full", 2025)
    runs = []
    for variant in VARIANTS:
        for seed in base.seeds:
            cfg = get_config(args.dataset, variant, seed)
            runs.append({
                "variant": variant,
                "seed": seed,
                "command": materialize_command(cfg, output_root, epochs=args.epochs),
            })
    if args.dry_run:
        print(json.dumps({
            "dataset": args.dataset, "selection": "peak_test", "runs": runs
        }, indent=2))
        return 0
    for item in runs:
        cfg = get_config(args.dataset, item["variant"], item["seed"])
        execute(cfg, output_root, epochs=args.epochs)
    return 0
