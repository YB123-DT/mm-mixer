from __future__ import annotations

import math


def warmup_cosine_decay(step: int, total_steps: int, warmup_steps: int) -> float:
    """Linear warmup followed by a single monotonic half-cosine decay."""
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must satisfy 0 <= warmup_steps < total_steps")
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))
