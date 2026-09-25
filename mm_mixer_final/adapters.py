from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import FinalConfig


def materialize_command(
    cfg: FinalConfig, output_root: Path, *, epochs: int | None = None
) -> list[str]:
    epochs = cfg.epochs if epochs is None else int(epochs)
    run_root = output_root / cfg.dataset / cfg.variant / f"seed{cfg.seed}"
    return [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / f"dataset_runners/{cfg.dataset}.py"),
        "--variant", cfg.variant, "--seed", str(cfg.seed),
        "--epochs", str(epochs), "--output-root", str(run_root),
    ]


def execute(cfg: FinalConfig, output_root: Path, *, epochs: int | None = None) -> None:
    subprocess.run(materialize_command(cfg, output_root, epochs=epochs), check=True)
