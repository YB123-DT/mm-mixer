from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import build_structured_cross_model


ROOT = Path("/data2/yb/multimodalERC/IEMOCAP/Model_rawaux_utterance_history_residual")
RUN = ROOT / "formal/structured_cross_screen/runs/X1"


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()), "std": float(array.std()),
        "p10": float(np.quantile(array, .1)), "median": float(np.median(array)),
        "p90": float(np.quantile(array, .9)), "max": float(array.max()),
    }


def main():
    import sys
    sys.path.insert(0, "/data2/yb/multimodalERC/IEMOCAP/Model_rawaux_fill53_css_aligned")
    from fill53_dataset import Fill53Dataset

    config = json.loads((RUN / "config.json").read_text())
    base = json.loads(Path(config["baseline_config"]).read_text())["fixed_params"]
    dropout = float(base.get("fusion_dropout", base.get("dropout_rate", .15)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_structured_cross_model("X1", dropout).to(device)
    model.load_state_dict(torch.load(RUN / "best_peak_test_state_dict.pt", map_location=device, weights_only=True), strict=True)
    model.eval()
    dataset = Fill53Dataset(config["pkl"], config["features"], "test", False)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=dataset.collate_fn)

    captured = {}
    handles = [
        model.feature_integrator.base_integrator.register_forward_hook(
            lambda _m, _i, output: captured.__setitem__("main", output.detach())
        ),
        model.transformer_encoder.cross.register_forward_hook(
            lambda _m, _i, output: captured.__setitem__("cross", output.detach())
        ),
    ]
    main_norms, cross_norms, ratios, cosines, logit_shifts = [], [], [], [], []
    correct_ratios, wrong_ratios = [], []
    with torch.no_grad():
        for features, labels in loader:
            features = {key: value.to(device) for key, value in features.items()}
            labels = labels.to(device)
            logits = model(features)[0]
            main, cross = captured["main"], captured["cross"]
            main_logits = torch.stack([head(main) for head in model.classifiers]).mean(0)
            n_main = main.norm(dim=-1); n_cross = cross.norm(dim=-1)
            ratio = n_cross / n_main.clamp_min(1e-8)
            cosine = F.cosine_similarity(main, cross, dim=-1)
            shift = (logits - main_logits).norm(dim=-1)
            is_correct = logits.argmax(-1).eq(labels)
            main_norms.extend(n_main.cpu().tolist()); cross_norms.extend(n_cross.cpu().tolist())
            ratios.extend(ratio.cpu().tolist()); cosines.extend(cosine.cpu().tolist())
            logit_shifts.extend(shift.cpu().tolist())
            correct_ratios.extend(ratio[is_correct].cpu().tolist())
            wrong_ratios.extend(ratio[~is_correct].cpu().tolist())
    for handle in handles: handle.remove()
    report = {
        "samples": len(ratios), "main_norm": summarize(main_norms),
        "cross_norm": summarize(cross_norms), "cross_to_main_ratio": summarize(ratios),
        "cosine_main_cross": summarize(cosines), "logit_shift_l2": summarize(logit_shifts),
        "ratio_correct": summarize(correct_ratios), "ratio_wrong": summarize(wrong_ratios),
    }
    (RUN / "residual_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
