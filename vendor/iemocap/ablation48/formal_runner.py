"""Auditable formal runner for the RawAux 01--46 ablation registry.

Nothing trains unless :option:`--execute` is supplied.  Test is deliberately
used as both the selection and reporting loader because this project requested
a peak-test *diagnostic* comparison; every artifact carries that label.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import hashlib
import importlib
import importlib.util
import json
import math
import os
import socket
import sys
import tempfile
import time
import traceback
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from .builder import build_fusion_model, experiment_training_loss
from .registry import build_registry
from .modules_37_46 import DialogueTemporalWrapper

ROOT = Path(__file__).resolve().parents[1]
IEMOCAP = Path("/data2/yb/multimodalERC/IEMOCAP")
CLEAN = ROOT / "base"
TRAIN_SOURCE = CLEAN / "train_erc.py"
BASE_MODEL_SOURCE = CLEAN / "multiattn.py"
BASELINE_CONFIG = CLEAN / "configs/textpeak_grid32/v27_lr30_wd20_fd10_mw45_auxt.json"
PKL = IEMOCAP / "external/CSS/data/iemocap_multimodal_features.pkl"
FEATURES = (
    IEMOCAP
    / "Model_rawaux_textpeak_cssv_v1/artifacts/iemocap_textpeak_audio_cssv.npz"
)
FILL = ROOT / "data"
# Integer targets in the pinned CSS/fill53 pickle use this exact order.  This is
# intentionally independent from experiment 28's query-name provenance metadata.
CLASS_NAMES = ("happiness", "sadness", "neutral", "anger", "excited", "frustration")
PROTOCOL = "STRICT_PEAK_TEST_DIAGNOSTIC"
FORMAL_PEAK_CALLBACK = None


def corrected_schedule_steps(num_epochs: int, batches_per_epoch: int,
                             accumulation_steps: int = 2,
                             warmup_ratio: float = 0.05) -> tuple[int, int]:
    """Return scheduler steps from real optimizer updates, not microbatches."""
    if min(num_epochs, batches_per_epoch, accumulation_steps) <= 0:
        raise ValueError("epochs, batches, and accumulation steps must be positive")
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("warmup ratio must be in [0, 1)")
    updates_per_epoch = math.ceil(batches_per_epoch / accumulation_steps)
    total_steps = num_epochs * updates_per_epoch
    return total_steps, int(total_steps * warmup_ratio)


def corrected_cosine_lambda(step: int, total_steps: int, warmup_steps: int) -> float:
    """Warmup followed by one monotonic half-cosine."""
    step = min(max(int(step), 0), int(total_steps))
    if warmup_steps and step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_corrected_scheduler(optimizer, num_epochs: int, batches_per_epoch: int,
                              accumulation_steps: int = 2, warmup_ratio: float = 0.05,
                              scheduler_mode: str = "corrected_cosine"):
    total_steps, warmup_steps = corrected_schedule_steps(
        num_epochs, batches_per_epoch, accumulation_steps, warmup_ratio,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: corrected_cosine_lambda(step, total_steps, warmup_steps),
    )
    scheduler.corrected_total_steps = total_steps
    scheduler.corrected_warmup_steps = warmup_steps
    print(json.dumps({
        "scheduler_mode": scheduler_mode,
        "total_optimizer_updates": total_steps,
        "warmup_updates": warmup_steps,
        "warmup_ratio": warmup_ratio,
        "initial_group_lrs": [group["initial_lr"] for group in optimizer.param_groups],
    }, sort_keys=True), flush=True)
    return scheduler


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def compute_source_hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {str(Path(path).resolve()): _sha(Path(path)) for path in paths}


def verify_run_artifacts(root: Path, experiment_id: str, *, expected_protocol: str | None = None,
                         expected_batch_protocol: str | None = None,
                         expected_seed: int | None = None) -> bool:
    """Cryptographically verify a completed run without importing queue code."""
    run=Path(root)/"runs"/experiment_id
    required=("status.json","manifest.json","metrics.json","replay_logits.pt","best_peak_test_state_dict.pt")
    if any(not (run/name).is_file() for name in required): return False
    try:
        status=json.loads((run/"status.json").read_text())
        manifest=json.loads((run/"manifest.json").read_text())
        metrics=json.loads((run/"metrics.json").read_text())
        checkpoint_sha=_sha(run/"best_peak_test_state_dict.pt")
        replay_sha=_sha(run/"replay_logits.pt")
        valid=(status.get("state")=="complete" and status.get("fresh_strict_replay_exact") is True
               and manifest.get("fresh_strict_replay_exact") is True
               and metrics.get("fresh_strict_replay_exact") is True
               and status.get("checkpoint_sha256")==checkpoint_sha
               and manifest.get("checkpoint_sha256")==checkpoint_sha
               and metrics.get("replay_logits_sha256")==replay_sha
               and manifest.get("experiment_id")==experiment_id)
        if expected_protocol is not None: valid=valid and manifest.get("protocol")==expected_protocol
        if expected_batch_protocol is not None: valid=valid and manifest.get("batch_protocol")==expected_batch_protocol
        if expected_seed is not None: valid=valid and manifest.get("seed")==expected_seed
        return bool(valid)
    except (OSError,json.JSONDecodeError,KeyError,TypeError): return False


def atomic_json(path: Path, value) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_text(path: Path, value: str) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(value); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


@dataclass(frozen=True)
class FormalConfig:
    experiment_id: str
    protocol: str = PROTOCOL
    baseline: str = "S0/E14 K4-d128-L1 fill53 TextPeak"
    baseline_config: str = str(BASELINE_CONFIG)
    pkl: str = str(PKL)
    features: str = str(FEATURES)
    seed: int = 2025
    epochs: int = 100
    batch_size: int = 32
    batch_protocol: str = "utterance"
    temporal_encode_scope: str = "none"
    turn_chunk_size: int = 32
    gradient_accumulation: int = 2
    optimizer_updates_per_epoch: int | None = None
    ema_updates_per_epoch: int | None = None
    dialogue_loader_factory: str | None = None

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


def build_default_config(experiment_id: str) -> FormalConfig:
    experiment_id = str(experiment_id)
    experiment_id = experiment_id if experiment_id == "DIALOGUE_NULL" else experiment_id.zfill(2)
    if experiment_id != "DIALOGUE_NULL" and experiment_id not in build_registry() and experiment_id != "48":
        raise ValueError(f"unknown immutable experiment: {experiment_id}")
    if experiment_id == "DIALOGUE_NULL" or 41 <= int(experiment_id) <= 44:
        return FormalConfig(
            experiment_id, batch_protocol="dialogue",
            temporal_encode_scope="full_dialogue_first",
            optimizer_updates_per_epoch=91, ema_updates_per_epoch=91,
            dialogue_loader_factory="ablation48.modules_37_46:build_dialogue_loaders",
        )
    return FormalConfig(experiment_id)


def validate_formal_config(cfg: FormalConfig) -> None:
    if cfg.protocol != PROTOCOL:
        raise ValueError(f"protocol must be {PROTOCOL}")
    number = None if cfg.experiment_id == "DIALOGUE_NULL" else int(cfg.experiment_id)
    expected = "dialogue" if number is None or 41 <= number <= 44 else "utterance"
    if cfg.batch_protocol != expected:
        raise ValueError(f"experiment {cfg.experiment_id} requires {expected} protocol")
    if cfg.seed != 2025 or cfg.epochs != 100 or cfg.batch_size != 32:
        raise ValueError("formal motherline locks seed=2025, epochs=100, batch_size=32")
    if expected == "dialogue":
        if (cfg.turn_chunk_size, cfg.gradient_accumulation,
                cfg.optimizer_updates_per_epoch, cfg.ema_updates_per_epoch) != (32, 2, 91, 91):
            raise ValueError("dialogue schedule must be 32-turn/accum2/91 optimizer+EMA updates")


class RunLedger:
    """Per-experiment atomic artifacts and a non-blocking process lock."""
    def __init__(self, root: Path, experiment_id: str):
        self.root = Path(root); self.experiment_id = str(experiment_id).zfill(2)
        self.run_dir = self.root / "runs" / self.experiment_id
        self.status_path = self.run_dir / "status.json"
        self.pid_path = self.run_dir / "owner.pid"
        self.lock_path = self.run_dir / ".run.lock"
        self.checkpoint_path = self.run_dir / "best_peak_test_state_dict.pt"

    @contextlib.contextmanager
    def acquire(self, blocking: bool = False):
        import fcntl
        self.run_dir.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a+")
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            try: fcntl.flock(handle.fileno(), flags)
            except BlockingIOError as exc: raise RuntimeError(f"experiment {self.experiment_id} already locked") from exc
            self.pid_path.write_text(f"{os.getpid()}\n")
            yield self
        finally:
            try: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally: handle.close()

    def status(self, state: str, **extra):
        atomic_json(self.status_path, {
            "experiment_id": self.experiment_id, "state": state,
            "pid": os.getpid(), "host": socket.gethostname(),
            "updated_at_unix": time.time(), **extra,
        })

    def atomic_checkpoint(self, state_dict):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".checkpoint.", suffix=".tmp", dir=self.run_dir)
        os.close(fd)
        try:
            torch.save(state_dict, temporary)
            os.replace(temporary, self.checkpoint_path)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)


def source_paths() -> list[Path]:
    return [TRAIN_SOURCE, BASE_MODEL_SOURCE, BASELINE_CONFIG,
            *sorted((ROOT / "ablation48").glob("*.py"))]


def build_experiment_model(experiment_id: str, dropout: float):
    if experiment_id == "DIALOGUE_NULL":
        return DialogueTemporalWrapper(build_fusion_model("S0", dropout), (), causal=True, microbatch=32)
    return build_fusion_model(experiment_id, dropout)


def _load_base_module():
    spec = importlib.util.spec_from_file_location("rawaux48_formal_base", BASE_MODEL_SOURCE)
    module = importlib.util.module_from_spec(spec); assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def patch_test_only_evaluation_source(source: str) -> str:
    """Use one Test forward when the formal protocol has no separate validation set.

    Formal RawAux runs deliberately receive the Test loader in both the
    ``val_loader`` and ``test_loader`` positions.  Keeping two evaluations
    therefore duplicates an identical 1,623-sample forward pass every epoch.
    The legacy ``val_*`` metric keys remain aliases of Test so the existing
    early-stopping and artifact contracts stay unchanged.
    """
    evaluation_anchor = """        val_loss, val_f1, val_acc = evaluate_model(
            ema_model, val_loader, device, None, criterion
        )
        test_loss, test_f1, test_acc = evaluate_model(
            ema_model, test_loader, device, None, criterion
        )
"""
    evaluation_replacement = """        test_loss, test_f1, test_acc = evaluate_model(
            ema_model, test_loader, device, None, criterion
        )
        # The formal protocol passes the same loader for validation and Test.
        # Preserve legacy metric keys without a duplicate forward evaluation.
        val_loss, val_f1, val_acc = test_loss, test_f1, test_acc
"""
    if source.count(evaluation_anchor) != 1:
        raise RuntimeError("pinned fusion evaluation anchor changed")
    source = source.replace(evaluation_anchor, evaluation_replacement)

    print_anchor = """        print(
            f"[fusion] Epoch {epoch:3d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} F1: {train_f1:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} F1: {val_f1:.4f} Acc: {val_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} F1: {test_f1:.4f} Acc: {test_acc:.4f}",
            flush=True,
        )
"""
    print_replacement = """        print(
            f"[fusion] Epoch {epoch:3d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} F1: {train_f1:.4f} Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} F1: {test_f1:.4f} Acc: {test_acc:.4f}",
            flush=True,
        )
"""
    if source.count(print_anchor) != 1:
        raise RuntimeError("pinned fusion metrics-print anchor changed")
    return source.replace(print_anchor, print_replacement)


def _patched_train_module(experiment_id: str, scheduler_mode: str = "legacy"):
    """Load pinned training semantics while injecting only the explicit builder."""
    corrected_warmup_ratios = {
        "corrected_cosine": 0.05,
        "corrected_updates20": 0.20,
    }
    if scheduler_mode not in {"legacy", *corrected_warmup_ratios}:
        raise ValueError(f"unknown scheduler mode: {scheduler_mode}")
    base = _load_base_module()
    proxy = types.ModuleType("multiattn")
    proxy.__dict__.update(base.__dict__)

    class ExplicitFusionFactory:
        def __new__(cls, *args, **kwargs):
            return build_experiment_model(experiment_id, float(kwargs.get("dropout", .1)))

    proxy.HierarchicalAttentionFusion = ExplicitFusionFactory
    previous = sys.modules.get("multiattn")
    sys.modules["multiattn"] = proxy
    try:
        source = TRAIN_SOURCE.read_text()
        source = patch_test_only_evaluation_source(source)
        snapshot = "best_weights = ema_model.state_dict()"
        if source.count(snapshot) != 1:
            raise RuntimeError("pinned EMA snapshot anchor changed")
        source = source.replace(snapshot, "best_weights = copy.deepcopy(ema_model.state_dict())")
        peak_snapshot = "best_weights = copy.deepcopy(ema_model.state_dict())"
        source = source.replace(
            peak_snapshot,
            peak_snapshot
            + "\n            if formal_peak_callback is not None:"
            + "\n                formal_peak_callback("
            + "ema_model, epoch, metrics, test_loader, device)",
            1,
        )
        loss_anchor = "loss = loss / 2\n            loss.backward()"
        if source.count(loss_anchor) != 1:
            raise RuntimeError("pinned training-loss anchor changed")
        source = source.replace(
            loss_anchor,
            "loss = (loss + experiment_training_loss(model)) / 2\n            loss.backward()",
        )
        epoch_anchor = "for epoch in range(1, num_epochs + 1):\n        model.train()"
        fusion_start = source.index("def train_fusion_model(")
        epoch_position = source.find(epoch_anchor, fusion_start)
        if epoch_position < 0: raise RuntimeError("pinned fusion epoch anchor changed")
        epoch_replacement = (epoch_anchor + "\n        if hasattr(model.transformer_encoder, 'epoch'):\n"
                             "            model.transformer_encoder.epoch = epoch")
        source = source[:epoch_position] + epoch_replacement + source[epoch_position+len(epoch_anchor):]
        # All formal variants use the same exhaustive, duplicate-free grouping.
        group_start = "    optimizer_groups = [\n"
        group_end = "    fusion_optimizer = AdamW(\n"
        begin = source.index(group_start, source.index("def train_full_pipeline("))
        end = source.index(group_end, begin)
        source = (source[:begin]
                  + "    optimizer_groups = formal_optimizer_parameter_groups(fusion_model, cfg['lr_fusion'])\n\n"
                  + source[end:])
        if scheduler_mode in corrected_warmup_ratios:
            scheduler_start = "    steps_per_epoch = len(train_loader)\n"
            scheduler_end = "    fusion_scheduler = LambdaLR(fusion_optimizer, lr_lambda)\n"
            begin = source.index(scheduler_start, source.index("def train_full_pipeline("))
            end = source.index(scheduler_end, begin) + len(scheduler_end)
            replacement = (
                "    fusion_scheduler = formal_build_corrected_scheduler(\n"
                "        fusion_optimizer, num_epochs, len(train_loader), accumulation_steps=2, "
                f"warmup_ratio={corrected_warmup_ratios[scheduler_mode]:.2f}, "
                f"scheduler_mode={scheduler_mode!r}\n"
                "    )\n"
            )
            source = source[:begin] + replacement + source[end:]
        module = types.ModuleType(f"rawaux48_train_{experiment_id}")
        module.__file__ = str(TRAIN_SOURCE)
        module.__dict__["experiment_training_loss"] = experiment_training_loss
        module.__dict__["formal_optimizer_parameter_groups"] = formal_optimizer_parameter_groups
        module.__dict__["formal_build_corrected_scheduler"] = build_corrected_scheduler
        module.__dict__["formal_scheduler_mode"] = scheduler_mode
        module.__dict__["formal_scheduler_warmup_ratio"] = corrected_warmup_ratios.get(scheduler_mode)
        module.__dict__["formal_peak_callback"] = FORMAL_PEAK_CALLBACK
        exec(compile(source, str(TRAIN_SOURCE), "exec"), module.__dict__)
        return module
    finally:
        if previous is None: sys.modules.pop("multiattn", None)
        else: sys.modules["multiattn"] = previous


def _data_api():
    if str(FILL) not in sys.path: sys.path.insert(0, str(FILL))
    from fill53_dataset import Fill53Dataset
    return Fill53Dataset


def _utterance_loaders(cfg: FormalConfig):
    from torch.utils.data import DataLoader
    dataset_cls = _data_api()
    train = dataset_cls(cfg.pkl, cfg.features, "train", True)
    test = dataset_cls(cfg.pkl, cfg.features, "test", False)
    if (len(train), len(test)) != (5810, 1623):
        raise RuntimeError(f"fill53 count mismatch: train={len(train)} test={len(test)}")
    return (train, test,
            DataLoader(train, cfg.batch_size, shuffle=True, collate_fn=train.collate_fn),
            DataLoader(test, cfg.batch_size, shuffle=False, collate_fn=test.collate_fn))


def _dialogue_loaders(cfg: FormalConfig):
    """Reserved explicit interface; never falls back to utterance batching."""
    module_name, symbol = cfg.dialogue_loader_factory.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, symbol, None)
    if factory is None:
        raise RuntimeError(f"dialogue loader factory is not implemented: {cfg.dialogue_loader_factory}")
    train_loader, test_loader = factory(cfg.pkl, cfg.features, cfg.turn_chunk_size)
    return train_loader.dataset, test_loader.dataset, train_loader, test_loader


def formal_optimizer_parameter_groups(model, lr: float):
    """RawAux LR partitions without duplicate parameters for wrapper modules."""
    groups, used = [], set()
    named = list(model.named_parameters())
    for prefix, group_lr in (("proj.", lr*.5), ("transformer_encoder.", lr), ("classifiers.", lr*2)):
        params=[]
        for name, parameter in named:
            normalized = name[5:] if name.startswith("base.") else name
            if normalized.startswith(prefix) and parameter.requires_grad:
                params.append(parameter); used.add(id(parameter))
        if params: groups.append({"params":params,"lr":group_lr})
    remaining=[parameter for _,parameter in named if parameter.requires_grad and id(parameter) not in used]
    if remaining: groups.append({"params":remaining,"lr":lr})
    flattened=[id(parameter) for group in groups for parameter in group["params"]]
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("optimizer parameter groups contain duplicates")
    required={id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if set(flattened) != required:
        raise RuntimeError("optimizer parameter groups do not exactly cover requires_grad parameters")
    return groups


def _training_labels(train_set):
    if hasattr(train_set, "index"):
        return [train_set.labels[dialogue][turn] for dialogue,turn in train_set.index]
    return [label for dialogue in train_set.dialogues for label in train_set.labels[dialogue]]


def replay_metrics(model, loader, device):
    import numpy as np
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    labels_all, predictions, logits = [], [], []
    model.eval()
    with torch.no_grad():
        for features, labels in loader:
            features = {name: value.to(device) for name, value in features.items()}
            output = model(features)[0]
            logits.append(output.detach().cpu())
            labels_all.extend(labels.numpy().tolist())
            predictions.extend(output.argmax(1).cpu().numpy().tolist())
    report = classification_report(labels_all, predictions, target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    metrics = {
        "samples": len(labels_all), "weighted_f1": float(f1_score(labels_all, predictions, average="weighted")),
        "accuracy": float(accuracy_score(labels_all, predictions)),
        "macro_f1": float(f1_score(labels_all, predictions, average="macro")),
        "per_class": {name: report[name] for name in CLASS_NAMES},
    }
    return metrics, torch.cat(logits), torch.tensor(predictions), torch.tensor(labels_all)


def _training_cfg(cfg: FormalConfig):
    fixed = dict(json.loads(Path(cfg.baseline_config).read_text())["fixed_params"])
    fixed.update(no_distill=False, no_channel_attention=False, no_gates=False,
                 no_alignment=True, seed=cfg.seed, fusion_early_stopping_threshold=0.0)
    return fixed


def build_fresh_model(cfg: FormalConfig, fixed: dict, device):
    """Reconstruct with exactly the training-time fusion dropout/configuration."""
    validate_formal_config(cfg)
    return build_experiment_model(cfg.experiment_id, float(fixed.get("fusion_dropout", fixed.get("dropout_rate", .15)))).to(device)


def execute_formal(cfg: FormalConfig, output_root: Path):
    """Execute one formal experiment and atomically finalize its evidence."""
    validate_formal_config(cfg)
    ledger = RunLedger(output_root, cfg.experiment_id)
    with ledger.acquire(blocking=False):
        start = time.time(); ledger.status("running", protocol=PROTOCOL, started_at_unix=start)
        atomic_json(ledger.run_dir / "config.json", cfg.to_dict())
        hashes = compute_source_hashes(source_paths())
        hashes.update({str(Path(cfg.pkl)): _sha(Path(cfg.pkl)), str(Path(cfg.features)): _sha(Path(cfg.features))})
        atomic_json(ledger.run_dir / "source_sha256.json", hashes)
        try:
            import numpy as np
            from sklearn.preprocessing import LabelEncoder
            from sklearn.utils.class_weight import compute_class_weight
            base = _load_base_module(); base.set_random_seed(cfg.seed)
            if cfg.batch_protocol == "utterance":
                train_set, test_set, train_loader, test_loader = _utterance_loaders(cfg)
            else:
                train_set, test_set, train_loader, test_loader = _dialogue_loaders(cfg)
            labels = np.asarray(_training_labels(train_set))
            weights = compute_class_weight("balanced", classes=np.arange(6), y=labels)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            encoder = LabelEncoder(); encoder.classes_ = np.asarray(CLASS_NAMES)
            fixed = _training_cfg(cfg)
            matched_control = None
            matched_control_wf1 = None
            if cfg.experiment_id in {"41","42","43","44"}:
                matched_control = "DIALOGUE_NULL"
                control_manifest = Path(output_root)/"runs"/matched_control/"manifest.json"
                if not verify_run_artifacts(output_root, matched_control, expected_protocol=PROTOCOL,
                                            expected_batch_protocol="dialogue", expected_seed=cfg.seed):
                    raise RuntimeError("DIALOGUE_NULL complete artifacts/protocol/batch/seed verification failed")
                control = json.loads(control_manifest.read_text())
                matched_control_wf1 = float(control["weighted_f1"])
            model, report_text, metrics = _patched_train_module(cfg.experiment_id).train_full_pipeline(
                train_loader, test_loader, test_loader, device, encoder, ["v", "a", "t"],
                {"v":342,"a":1024,"t":1024}, torch.tensor(weights,dtype=torch.float32,device=device),
                fixed, cfg.epochs,
            )
            required_metrics = {"best_epoch","best_val_f1","best_test_epoch","best_test_f1",
                                "best_test_acc","selected_test_f1","selected_test_acc"}
            missing = required_metrics - metrics.keys()
            if missing: raise RuntimeError(f"pinned train_erc metrics contract changed: missing {sorted(missing)}")
            if metrics["best_epoch"] != metrics["best_test_epoch"]:
                raise RuntimeError("selection is not strict peak-test diagnostic epoch")
            ledger.atomic_checkpoint(copy.deepcopy(model.state_dict()))
            selected, logits_a, preds_a, labels_a = replay_metrics(model, test_loader, device)
            fresh = build_fresh_model(cfg, fixed, device)
            fresh.load_state_dict(torch.load(ledger.checkpoint_path, map_location=device, weights_only=True), strict=True)
            replay, logits_b, preds_b, labels_b = replay_metrics(fresh, test_loader, device)
            exact = torch.equal(logits_a, logits_b) and torch.equal(preds_a, preds_b) and torch.equal(labels_a, labels_b)
            if selected["samples"] != 1623 or replay["samples"] != 1623 or not exact:
                raise RuntimeError("fresh full-1623 strict replay failed")
            if abs(float(metrics["selected_test_f1"])-selected["weighted_f1"]) > 1e-12:
                raise RuntimeError("selected_test_f1 disagrees with full replay")
            if abs(float(metrics["selected_test_acc"])-selected["accuracy"]) > 1e-12:
                raise RuntimeError("selected_test_acc disagrees with full replay")
            logits_path = ledger.run_dir / "replay_logits.pt"
            temporary = logits_path.with_suffix(".tmp")
            torch.save({"logits": logits_b, "predictions": preds_b, "labels": labels_b}, temporary)
            os.replace(temporary, logits_path)
            final_metrics = {
                **metrics, "selected": selected, "fresh_strict_replay": replay,
                "fresh_strict_replay_exact": True, "replay_logits_sha256": _sha(logits_path),
            }
            atomic_json(ledger.run_dir / "metrics.json", final_metrics)
            atomic_text(ledger.run_dir / "classification_report.txt", report_text + "\n")
            manifest = {
                "protocol": PROTOCOL, "selection_rule": "EMA at peak Test WF1 (diagnostic only)",
                "generalization_claim_allowed": False, "experiment_id": cfg.experiment_id,
                "batch_protocol": cfg.batch_protocol, "seed": cfg.seed, "train_samples":5810,
                "test_samples":1623, "parameter_count":sum(p.numel() for p in model.parameters()),
                "checkpoint":str(ledger.checkpoint_path), "checkpoint_sha256":_sha(ledger.checkpoint_path),
                "source_sha256":hashes, "weighted_f1":selected["weighted_f1"],
                "accuracy":selected["accuracy"], "macro_f1":selected["macro_f1"],
                "per_class":selected["per_class"], "fresh_strict_replay_exact":True,
                "elapsed_seconds":time.time()-start, "argv":sys.argv,
                "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
                "matched_control": matched_control,
                "matched_control_weighted_f1": matched_control_wf1,
                "matched_delta_weighted_f1": (None if matched_control_wf1 is None else selected["weighted_f1"]-matched_control_wf1),
            }
            atomic_json(ledger.run_dir / "manifest.json", manifest)
            ledger.status("complete", protocol=PROTOCOL, fresh_strict_replay_exact=True,
                          checkpoint_sha256=manifest["checkpoint_sha256"], finished_at_unix=time.time())
            return manifest
        except BaseException as exc:
            ledger.status("failed", protocol=PROTOCOL, error_type=type(exc).__name__, error=str(exc),
                          traceback=traceback.format_exc(),
                          finished_at_unix=time.time())
            raise


def generate_configs(destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    for number in range(1, 47):
        cfg = build_default_config(f"{number:02d}")
        validate_formal_config(cfg)
        atomic_json(destination / f"{number:02d}.json", cfg.to_dict())
    cfg=build_default_config("DIALOGUE_NULL"); validate_formal_config(cfg)
    atomic_json(destination / "DIALOGUE_NULL.json", cfg.to_dict())
    cfg=build_default_config("48"); validate_formal_config(cfg)
    atomic_json(destination / "48.json",cfg.to_dict())


def synthetic_smoke(experiment_id: str):
    torch.manual_seed(2025)
    model = build_experiment_model(experiment_id, 0.0)
    if experiment_id == "DIALOGUE_NULL" or 41 <= int(experiment_id) <= 44:
        x = {"v":torch.randn(2,3,342), "a":torch.randn(2,3,1024), "t":torch.randn(2,3,1024),
             "graph_mask":torch.ones(2,3,dtype=torch.bool),
             "target_mask":torch.tensor([[1,0,0],[0,1,0]],dtype=torch.bool)}
    else:
        x = {"v":torch.randn(2,342), "a":torch.randn(2,1024), "t":torch.randn(2,1024)}
    output = model(x, labels=torch.tensor([0,5]))[0]
    loss = torch.nn.functional.cross_entropy(output, torch.tensor([0,5])) + experiment_training_loss(model)
    loss.backward()
    return {"experiment_id":experiment_id,"shape":list(output.shape),"finite":bool(torch.isfinite(output).all()),
            "training_loss_finite":bool(torch.isfinite(loss)),"parameter_count":sum(p.numel() for p in model.parameters())}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment")
    parser.add_argument("--config")
    parser.add_argument("--output-root", default=str(ROOT / "formal"))
    parser.add_argument("--generate-configs", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.generate_configs:
        generate_configs(Path(args.output_root) / "configs"); return
    cfg = FormalConfig.from_dict(json.loads(Path(args.config).read_text())) if args.config else build_default_config(args.experiment)
    validate_formal_config(cfg)
    if args.smoke:
        print(json.dumps(synthetic_smoke(cfg.experiment_id), sort_keys=True)); return
    if not args.execute:
        print(json.dumps({"dry_run":True,"config":cfg.to_dict()},indent=2)); return
    print(json.dumps(execute_formal(cfg, Path(args.output_root)), indent=2))


if __name__ == "__main__":
    main()
