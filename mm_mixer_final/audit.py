from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Iterable

from torch import nn

from .config import FinalConfig


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(paths: Iterable[Path | str]) -> dict[str, str]:
    return {str(Path(path).resolve()): sha256(path) for path in paths}


def config_payload_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def loaded_python_source_hashes(
    final_root: Path | str | None = None,
    extra_paths: Iterable[Path | str] = (),
) -> dict[str, str]:
    root = (
        Path(final_root).resolve()
        if final_root is not None
        else Path(__file__).resolve().parents[1]
    )
    paths: set[Path] = {Path(value).resolve() for value in extra_paths}
    for module in tuple(sys.modules.values()):
        raw = getattr(module, "__file__", None)
        if not raw:
            continue
        path = Path(raw)
        if path.suffix in {".pyc", ".pyo"}:
            try:
                path = Path(importlib.util.source_from_cache(str(path)))
            except ValueError:
                continue
        try:
            resolved = path.resolve()
            resolved.relative_to(root)
        except (OSError, ValueError):
            continue
        if resolved.suffix == ".py" and resolved.is_file():
            paths.add(resolved)
    return source_hashes(sorted(paths))


def optimizer_group_audit(model: nn.Module, groups: Iterable[dict]) -> list[dict]:
    names_by_id = {
        id(parameter): name for name, parameter in model.named_parameters()
    }
    result = []
    for group in groups:
        names = [names_by_id[id(parameter)] for parameter in group["params"]]
        if names and all(name.startswith("proj.") for name in names):
            label = "projection"
        elif names and all(
            name.startswith("transformer_encoder.cross.") for name in names
        ):
            label = "pairwise_cross"
        elif names and all(name.startswith("cross_attn.") for name in names):
            label = "cross_attention"
        elif names and all(
            name.startswith("transformer_encoder.") for name in names
        ):
            label = "mixer_encoder"
        elif names and all(name.startswith("classifiers.") for name in names):
            label = "classifier"
        else:
            label = "remaining"
        result.append({
            "name": label,
            "lr": float(group["lr"]),
            "parameter_count": int(
                sum(parameter.numel() for parameter in group["params"])
            ),
            "parameter_names": names,
        })
    if len({item["name"] for item in result}) != len(result):
        raise RuntimeError("optimizer audit produced duplicate group names")
    return result


def assert_true_mixer(model: nn.Module, cfg: FinalConfig) -> None:
    encoder = model.transformer_encoder
    blocks = encoder.blocks
    assert len(blocks) == cfg.mixer["blocks"]
    assert encoder.token_count == cfg.mixer["tokens"]
    assert encoder.token_dim == cfg.mixer["dim"]
    for block in blocks:
        assert isinstance(block.route, nn.Linear)
        assert block.route.bias is None
        assert tuple(block.route.weight.shape) == (3, 3)
        assert block.ffn[0].out_features == cfg.mixer["ffn"]
