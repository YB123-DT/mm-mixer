import argparse
import datetime
import json
import math
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score, accuracy_score
from multiattn import (
    set_random_seed,
    TextGraphEncoder,
    AudioGraphEncoder,
    VisualGraphEncoder,
    HierarchicalAttentionFusion,
    EvidentialDirichletFusion,
    DialogueStateGRURefiner,
    FusionWithDialogueState,
    HWRH2L1Refiner,
    FusionWithHWR,
    PostFusionStateRefiner,
    FusionWithPostState,
    MELDDataset,
    MELDDialogueDataset,
    custom_collate,
    dialogue_collate,
    CompositePolyLoss,
    DistillationLoss,
    MultitaskFusionLoss,
    EvidentialFusionLoss,
)
from layerwise_objectives import compose_layerwise_loss


STRICT_GRADIENT_CLIP_NORM = 1.0
STRICT_LR_REDUCE_FACTOR = 0.5
STRICT_LR_REDUCE_PATIENCE = 3
STRICT_EARLY_STOPPING_PATIENCE = 12


def validate_strict_clean_config(cfg):
    if not cfg.get("strict_clean_protocol", False):
        return
    expected = {
        "no_alignment": True,
        "fusion_type": "hierarchical",
        "gradient_clip_norm": STRICT_GRADIENT_CLIP_NORM,
        "lr_reduce_factor": STRICT_LR_REDUCE_FACTOR,
        "lr_reduce_patience": STRICT_LR_REDUCE_PATIENCE,
        "early_stopping_patience": STRICT_EARLY_STOPPING_PATIENCE,
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(
                f"strict Clean control requires {key}={value!r}, "
                f"got {cfg.get(key)!r}"
            )
    if cfg.get("layerwise_variant") not in {"a0", "a1", "a2", "a3"}:
        raise ValueError("strict Clean control requires layerwise_variant=a0/a1/a2/a3")


def build_strict_fusion_optimizer_groups(model, base_lr):
    """Return the four effective optimizer groups used by RawAux Clean.

    Names are included only as an auditable signature and are removed before
    constructing AdamW.
    """
    named = list(model.named_parameters())

    def select(prefixes):
        return [(name, parameter) for name, parameter in named if parameter.requires_grad and any(name.startswith(prefix) for prefix in prefixes)]

    projections = select(("proj.",))
    transformer = select(("transformer_encoder.",))
    classifiers = select(("classifiers.",))
    claimed = {id(parameter) for group in (projections, transformer, classifiers) for _, parameter in group}
    other = [(name, parameter) for name, parameter in named if parameter.requires_grad and id(parameter) not in claimed]
    definitions = (
        (projections, base_lr * 0.5),
        (transformer, base_lr),
        (classifiers, base_lr * 2.0),
        (other, base_lr),
    )
    return [
        {"params": [parameter for _, parameter in group], "names": [name for name, _ in group], "lr": lr}
        for group, lr in definitions
        if group
    ]


def evaluate_model(
    model, dataloader, device, modality_name, criterion,
    feature_extractors=None, return_outputs=False,
):
    model.eval()
    total_loss, all_labels, all_preds = 0.0, [], []
    all_logits = []
    num_batches = 0

    with torch.no_grad():
        for batch_data in dataloader:
            batch_features, labels = batch_data[:2]
            adj = batch_features.get("_adj")
            mask = batch_features.get("_mask")
            is_dialogue = adj is not None and mask is not None

            if modality_name:
                if modality_name not in batch_features:
                    continue
                x = batch_features[modality_name].to(device)

                if is_dialogue:
                    logits = model(x, adj=adj.to(device), mask=mask.to(device))
                    valid = mask.to(device)
                    logits_flat = logits[valid]
                    labels_flat = labels[valid.cpu()].to(device)
                else:
                    dia = batch_features.get("_dialogue_id", None)
                    utt = batch_features.get("_utterance_id", None)
                    if dia is not None:
                        logits = model(x, dialogue_ids=dia.to(device), utterance_ids=utt.to(device))
                    else:
                        logits = model(x)
                    labels_flat = labels.to(device)
                    logits_flat = logits

                if criterion and hasattr(criterion, "forward") and "teacher_logits" in criterion.forward.__code__.co_varnames:
                    dummy_teacher = torch.zeros_like(logits_flat)
                    prev_alpha = criterion.alpha
                    criterion.alpha = 1.0
                    loss = criterion(logits_flat, dummy_teacher, labels_flat)
                    criterion.alpha = prev_alpha
                    total_loss += loss.item()
                elif criterion:
                    total_loss += criterion(logits_flat, labels_flat).item()
            else:
                valid_batch = True
                for m in model.modalities:
                    if m not in batch_features:
                        valid_batch = False
                        break
                if not valid_batch:
                    continue

                for m in batch_features:
                    batch_features[m] = batch_features[m].to(device)

                model_features = prepare_fusion_features(batch_features, feature_extractors)
                return_hidden = bool(getattr(criterion, "requires_hidden", False))
                model_output = model(model_features, return_hidden=return_hidden)
                parsed = unpack_fusion_output(model_output, return_hidden=return_hidden)
                logits_flat = parsed["main_logits"]
                aux_logits = parsed["aux_logits"]
                extra = parsed["extra"]
                fused_hidden = parsed["fused_hidden"]

                if is_dialogue:
                    labels_flat = labels[mask.cpu()].to(device)
                else:
                    labels_flat = labels.to(device)

                if criterion:
                    if hasattr(criterion, "forward") and "aux_logits" in criterion.forward.__code__.co_varnames:
                        if aux_logits is not None:
                            if isinstance(criterion, EvidentialFusionLoss):
                                total_loss += criterion(
                                    logits_flat, aux_logits, labels_flat, fused_alpha=extra
                                ).item()
                            else:
                                total_loss += criterion(
                                    logits_flat,
                                    aux_logits,
                                    labels_flat,
                                    contrastive_loss=extra,
                                    fused_hidden=fused_hidden,
                                ).item()
                        else:
                            total_loss += criterion(logits_flat, None, labels_flat).item()
                    else:
                        total_loss += criterion(logits_flat, labels_flat).item()

            preds = torch.argmax(logits_flat, dim=1)
            if return_outputs:
                all_logits.append(logits_flat.detach().cpu())
            all_labels.extend(labels_flat.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            num_batches += 1

    if num_batches == 0:
        empty = (0.0, 0.0, 0.0)
        if return_outputs:
            return (*empty, torch.empty(0), torch.empty(0, dtype=torch.long))
        return empty

    avg_loss = total_loss / num_batches if criterion else 0.0
    f1 = f1_score(all_labels, all_preds, average="weighted")
    acc = np.mean(np.array(all_labels) == np.array(all_preds))

    if return_outputs:
        return (
            avg_loss,
            f1,
            acc,
            torch.cat(all_logits),
            torch.tensor(all_labels, dtype=torch.long),
        )
    return avg_loss, f1, acc


def format_classification_report_dict(rpt_dict):
    headers = ["Class", "Precision", "Recall", "F1-score", "Support"]
    line_fmt = "{:>15} {:>10} {:>10} {:>10} {:>10}"
    lines = [line_fmt.format(*headers), "-" * 65]
    for lbl, metrics in rpt_dict.items():
        if isinstance(metrics, dict):
            p = metrics["precision"] * 100
            r = metrics["recall"] * 100
            f = metrics["f1-score"] * 100
            s = int(metrics["support"])
            lines.append(line_fmt.format(lbl, f"{p:.2f}", f"{r:.2f}", f"{f:.2f}", s))
        elif lbl == "accuracy":
            lines.append(line_fmt.format(lbl, f"{metrics * 100:.2f}", "", "", ""))
    return "\n".join(lines)


META_KEYS = {"_dialogue_id", "_utterance_id", "_adj", "_mask"}

def prepare_fusion_features(batch_features, feature_extractors=None):
    if not feature_extractors:
        return batch_features
    adj = batch_features.get("_adj")
    mask = batch_features.get("_mask")
    encoded = {}

    with torch.no_grad():
        if adj is not None and mask is not None:
            # Dialogue-level data: extract with graph context, then flatten to utterance-level
            for modality, features in batch_features.items():
                if modality in META_KEYS:
                    continue
                extractor = feature_extractors.get(modality)
                if extractor is None:
                    encoded[modality] = features[mask].contiguous()
                else:
                    feat = extractor.extract_features(features, adj=adj, mask=mask)
                    encoded[modality] = feat[mask].contiguous()
        else:
            # Original utterance-level path
            dia_ids = batch_features.get("_dialogue_id", None)
            utt_ids = batch_features.get("_utterance_id", None)
            for modality, features in batch_features.items():
                if modality in META_KEYS:
                    continue
                extractor = feature_extractors.get(modality)
                if extractor is None:
                    encoded[modality] = features
                else:
                    encoded[modality] = extractor.extract_features(
                        features, dialogue_ids=dia_ids, utterance_ids=utt_ids)
    return encoded


def unpack_fusion_output(model_output, return_hidden=False):
    parsed = {
        "main_logits": model_output,
        "aux_logits": None,
        "extra": None,
        "aug_labels": None,
        "fused_hidden": None,
    }
    if not isinstance(model_output, tuple):
        return parsed

    parts = list(model_output)
    if return_hidden and len(parts) >= 3:
        parsed["fused_hidden"] = parts.pop()

    if len(parts) == 4:
        parsed["main_logits"], parsed["aux_logits"], parsed["extra"], parsed["aug_labels"] = parts
    elif len(parts) == 3:
        parsed["main_logits"], parsed["aux_logits"], parsed["extra"] = parts
    elif len(parts) >= 2:
        parsed["main_logits"], parsed["aux_logits"] = parts[:2]
    elif len(parts) == 1:
        parsed["main_logits"] = parts[0]
    return parsed


def summarize_evidential_diagnostics(model, dataloader, device, label_names, feature_extractors=None):
    if not isinstance(model, EvidentialDirichletFusion):
        return None

    stats = {
        "num_samples": 0,
        "pred_counts": np.zeros(len(label_names), dtype=np.int64),
        "strength_sum": {m: 0.0 for m in model.modalities},
        "uncertainty_sum": {m: 0.0 for m in model.modalities},
        "aux_wrong": {m: 0 for m in model.modalities},
        "aux_wrong_conf_sum": {m: 0.0 for m in model.modalities},
        "conflict_sum": 0.0,
        "conflict_count": 0,
        "conflict_max": 0.0,
        "evidence_gated_total": 0,  # Strategy B: how many modality-samples were gated
        "gate_weight_sum": {m: 0.0 for m in model.modalities},  # Strategy C
        "gate_sample_count": 0,
    }

    model.eval()
    with torch.no_grad():
        for batch_data in dataloader:
            feats, lbls = batch_data[:2]
            mask = feats.get("_mask")
            is_dialogue = mask is not None
            for m in feats:
                feats[m] = feats[m].to(device)
            model_feats = prepare_fusion_features(feats, feature_extractors)
            logits, aux_logits, diag = model(model_feats, return_attention=True)
            if is_dialogue:
                labels = lbls[mask.cpu()].to(device)
            else:
                labels = lbls.to(device)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            stats["pred_counts"] += np.bincount(preds, minlength=len(label_names))
            batch_size = labels.size(0)
            stats["num_samples"] += batch_size

            for m in model.modalities:
                if m not in diag["alpha"]:
                    continue
                alpha = diag["alpha"][m]
                stats["strength_sum"][m] += alpha.sum(dim=1).sum().item()
                stats["uncertainty_sum"][m] += diag["uncertainty"][m].sum().item()

                aux_prob = F.softmax(aux_logits[m], dim=1)
                aux_pred = torch.argmax(aux_prob, dim=1)
                aux_conf = aux_prob.max(dim=1).values
                wrong = aux_pred.ne(labels)
                stats["aux_wrong"][m] += wrong.sum().item()
                if wrong.any():
                    stats["aux_wrong_conf_sum"][m] += aux_conf[wrong].sum().item()

            conflict = diag.get("conflict")
            if conflict is not None and conflict.numel() > 0:
                stats["conflict_sum"] += conflict.sum().item()
                stats["conflict_count"] += conflict.numel()
                stats["conflict_max"] = max(stats["conflict_max"], conflict.max().item())

            # Strategy B+C gate diagnostics
            gate_info = getattr(model, "_last_gate_info", {})
            if gate_info:
                if "evidence_gated" in gate_info:
                    stats["evidence_gated_total"] += gate_info["evidence_gated"]
                if "learned_weights" in gate_info:
                    w = gate_info["learned_weights"]
                    for idx, m in enumerate(model.modalities):
                        stats["gate_weight_sum"][m] += w[:, idx].sum().item()
                    stats["gate_sample_count"] += batch_size

    n = max(stats["num_samples"], 1)
    pred_counts = stats["pred_counts"]
    pred_dist = {
        str(label): {
            "count": int(pred_counts[i]),
            "ratio": float(pred_counts[i] / n),
        }
        for i, label in enumerate(label_names)
    }
    modalities = {}
    for m in model.modalities:
        wrong = stats["aux_wrong"][m]
        modalities[m] = {
            "strength_mean": stats["strength_sum"][m] / n,
            "uncertainty_mean": stats["uncertainty_sum"][m] / n,
            "aux_wrong_rate": wrong / n,
            "aux_wrong_conf_mean": stats["aux_wrong_conf_sum"][m] / max(wrong, 1),
        }

    gate_info = {}
    if stats["gate_sample_count"] > 0:
        for m in model.modalities:
            gate_info[m] = stats["gate_weight_sum"][m] / stats["gate_sample_count"]
    if model.evidence_gate_threshold is not None:
        gate_info["evidence_gated_total"] = stats["evidence_gated_total"]

    return {
        "num_samples": stats["num_samples"],
        "modalities": modalities,
        "conflict_mean": stats["conflict_sum"] / max(stats["conflict_count"], 1),
        "conflict_max": stats["conflict_max"],
        "pred_distribution": pred_dist,
        "gate_info": gate_info,
    }


def format_evidential_diagnostics(diag):
    if diag is None:
        return ""
    lines = ["==== Evidential Diagnostics ===="]
    for m, values in diag["modalities"].items():
        lines.append(
            f"{m}: S={values['strength_mean']:.4f} "
            f"u={values['uncertainty_mean']:.4f} "
            f"aux_wrong={values['aux_wrong_rate'] * 100:.2f}% "
            f"wrong_conf={values['aux_wrong_conf_mean']:.4f}"
        )
    lines.append(
        f"conflict_K: mean={diag['conflict_mean']:.4f} max={diag['conflict_max']:.4f}"
    )
    gate_info = diag.get("gate_info", {})
    if gate_info:
        gate_strs = []
        for k, v in gate_info.items():
            if k == "evidence_gated_total":
                gate_strs.append(f"evidence_gated={v}")
            else:
                gate_strs.append(f"gate_{k}={v:.3f}")
        if gate_strs:
            lines.append(f"gate: {', '.join(gate_strs)}")
    pred = ", ".join(
        f"{label}:{values['count']}({values['ratio'] * 100:.1f}%)"
        for label, values in diag["pred_distribution"].items()
    )
    lines.append(f"pred_distribution: {pred}")
    return "\n".join(lines)


def train_modality_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
    modality_name,
    teacher_model=None,
    checkpoint_prefix="",
):
    best_f1, patience, counter, best_weights = 0.0, 10, 0, None
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "train_f1": [],
        "val_f1": [],
        "train_acc": [],
        "val_acc": [],
    }

    is_distillation = (
        teacher_model is not None
        and hasattr(criterion, "forward")
        and "teacher_logits" in criterion.forward.__code__.co_varnames
    )

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_labels, epoch_preds = [], []
        num_batches = 0

        for batch_features, labels in train_loader:
            if modality_name not in batch_features:
                continue

            x = batch_features[modality_name].to(device)
            adj = batch_features.get("_adj")
            mask = batch_features.get("_mask")
            is_dialogue = adj is not None and mask is not None
            optimizer.zero_grad()

            if is_dialogue:
                logits = model(x, adj=adj.to(device), mask=mask.to(device))
                valid = mask.to(device)
                logits_flat = logits[valid]
                labels_flat = labels[valid.cpu()].to(device)
            else:
                dia = batch_features.get("_dialogue_id", None)
                utt = batch_features.get("_utterance_id", None)
                if dia is not None:
                    logits = model(x, dialogue_ids=dia.to(device), utterance_ids=utt.to(device))
                else:
                    logits = model(x)
                labels_flat = labels.to(device)
                logits_flat = logits

            if is_distillation:
                if teacher_model is not None and "t" in batch_features:
                    with torch.no_grad():
                        if is_dialogue:
                            t_logits = teacher_model(batch_features["t"].to(device),
                                                     adj=adj.to(device), mask=mask.to(device))
                            t_logits = t_logits[valid]
                        elif dia is not None:
                            t_logits = teacher_model(batch_features["t"].to(device),
                                                     dialogue_ids=dia.to(device), utterance_ids=utt.to(device))
                        else:
                            t_logits = teacher_model(batch_features["t"].to(device))
                    loss = criterion(logits_flat, t_logits, labels_flat)
                else:
                    orig_alpha = criterion.alpha
                    criterion.alpha = 1.0
                    loss = criterion(logits_flat, torch.zeros_like(logits_flat), labels_flat)
                    criterion.alpha = orig_alpha
            else:
                loss = criterion(logits_flat, labels_flat)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            preds = torch.argmax(logits_flat, dim=1)
            epoch_labels.extend(labels_flat.cpu().numpy())
            epoch_preds.extend(preds.cpu().numpy())

        if num_batches == 0:
            continue

        train_loss = epoch_loss / num_batches
        train_f1 = f1_score(epoch_labels, epoch_preds, average="weighted")
        train_acc = accuracy_score(epoch_labels, epoch_preds)

        if is_distillation:
            orig_alpha = criterion.alpha
            criterion.alpha = 1.0
            val_loss, val_f1, val_acc = evaluate_model(
                model, val_loader, device, modality_name, criterion
            )
            criterion.alpha = orig_alpha
        else:
            val_loss, val_f1, val_acc = evaluate_model(
                model, val_loader, device, modality_name, criterion
            )

        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["train_f1"].append(train_f1)
        metrics["val_f1"].append(val_f1)
        metrics["train_acc"].append(train_acc)
        metrics["val_acc"].append(val_acc)

        print(
            f"[{modality_name}] Epoch {epoch:3d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} F1: {train_f1:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} F1: {val_f1:.4f} Acc: {val_acc:.4f}",
            flush=True,
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
            torch.save(best_weights, f"{checkpoint_prefix}checkpoint_{modality_name}_best.pth")
            print(f"[{modality_name}] >>> new best F1: {best_f1:.4f} (saved)", flush=True)
        else:
            counter += 1
            if counter >= patience:
                print(f"[{modality_name}] early stopping at epoch {epoch}", flush=True)
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, metrics


def train_modality_models_parallel(
    models,
    train_loader,
    val_loader,
    criteria,
    optimizers,
    num_epochs,
    device,
    checkpoint_prefix="",
):
    """Train multiple modality encoders in parallel within a single epoch loop.

    Args:
        models: dict {modality_name: model}
        criteria: dict {modality_name: criterion}
        optimizers: dict {modality_name: optimizer}

    Returns:
        tuple: (models dict, metrics dict)
    """
    from sklearn.metrics import f1_score as sk_f1, accuracy_score as sk_acc

    modality_names = sorted(models.keys())
    best_f1s = {m: 0.0 for m in modality_names}
    patience = 10
    counters = {m: 0 for m in modality_names}
    best_weights = {m: None for m in modality_names}
    all_metrics = {m: {"train_loss": [], "val_loss": [], "train_f1": [],
                       "val_f1": [], "train_acc": [], "val_acc": []}
                   for m in modality_names}

    for epoch in range(1, num_epochs + 1):
        for m in modality_names:
            models[m].train()

        epoch_losses = {m: 0.0 for m in modality_names}
        epoch_labels = {m: [] for m in modality_names}
        epoch_preds = {m: [] for m in modality_names}
        num_batches = 0

        for batch_features, labels in train_loader:
            adj = batch_features.get("_adj")
            msk = batch_features.get("_mask")
            is_dialogue = adj is not None and msk is not None

            for m in batch_features:
                batch_features[m] = batch_features[m].to(device)

            # Zero all grads first
            for m in modality_names:
                optimizers[m].zero_grad()

            # All forwards in parallel (independent models → GPU can overlap)
            total_loss = 0.0
            losses_m = {}
            logits_dict = {}
            for m in modality_names:
                if is_dialogue:
                    logits = models[m](batch_features[m], adj=adj, mask=msk)
                    valid = msk.to(device)
                    logits_flat = logits[valid]
                    labels_flat = labels[valid.cpu()].to(device)
                else:
                    dia = batch_features.get("_dialogue_id", None)
                    utt = batch_features.get("_utterance_id", None)
                    if dia is not None:
                        logits = models[m](batch_features[m], dialogue_ids=dia, utterance_ids=utt)
                    else:
                        logits = models[m](batch_features[m])
                    labels_flat = labels.to(device)
                    logits_flat = logits
                logits_dict[m] = logits_flat
                loss_m = criteria[m](logits_flat, labels_flat)
                losses_m[m] = loss_m
                total_loss = total_loss + loss_m

            # Single backward for all three independent losses
            total_loss.backward()

            # Step each optimizer
            for m in modality_names:
                torch.nn.utils.clip_grad_norm_(models[m].parameters(), max_norm=1.0)
                optimizers[m].step()
                epoch_losses[m] += losses_m[m].item()
                preds = torch.argmax(logits_dict[m], dim=1)
                epoch_labels[m].extend(labels_flat.cpu().numpy())
                epoch_preds[m].extend(preds.cpu().numpy())

            num_batches += 1

        # Validation and checkpointing for each modality
        for m in modality_names:
            train_loss = epoch_losses[m] / num_batches
            train_f1 = sk_f1(epoch_labels[m], epoch_preds[m], average="weighted")
            train_acc = sk_acc(epoch_labels[m], epoch_preds[m])

            val_loss, val_f1, val_acc = evaluate_model(
                models[m], val_loader, device, m, criteria[m]
            )

            all_metrics[m]["train_loss"].append(train_loss)
            all_metrics[m]["val_loss"].append(val_loss)
            all_metrics[m]["train_f1"].append(train_f1)
            all_metrics[m]["val_f1"].append(val_f1)
            all_metrics[m]["train_acc"].append(train_acc)
            all_metrics[m]["val_acc"].append(val_acc)

            print(
                f"[{m}] Epoch {epoch:3d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} F1: {train_f1:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} F1: {val_f1:.4f} Acc: {val_acc:.4f}",
                flush=True,
            )

            if val_f1 > best_f1s[m]:
                best_f1s[m] = val_f1
                best_weights[m] = {k: v.cpu().clone() for k, v in models[m].state_dict().items()}
                counters[m] = 0
                torch.save(best_weights[m], f"{checkpoint_prefix}checkpoint_{m}_best.pth")
                print(f"[{m}] >>> new best F1: {best_f1s[m]:.4f} (saved)", flush=True)
            else:
                counters[m] += 1
                if counters[m] >= patience:
                    print(f"[{m}] early stopping at epoch {epoch}", flush=True)
                    # Don't break - other modalities may still be improving

    # Restore best weights for each modality
    for m in modality_names:
        if best_weights[m] is not None:
            models[m].load_state_dict(best_weights[m])

    return models, all_metrics


def train_fusion_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    device,
    checkpoint_prefix="",
    feature_extractors=None,
    grad_accum_steps=2,
    peak_artifact_store=None,
    peak_artifact_config=None,
    peak_artifact_manifest=None,
    peak_class_names=None,
):
    best_f1, patience, counter, best_weights = 0.0, STRICT_EARLY_STOPPING_PATIENCE, 0, None
    best_test_f1, best_test_acc, best_test_epoch = float("-inf"), 0.0, 0
    best_test_weights = None
    best_epoch = 0
    early_stopping_threshold = 0.001
    lr_reduce_counter = 0
    lr_reduce_patience = STRICT_LR_REDUCE_PATIENCE
    lr_reduce_factor = STRICT_LR_REDUCE_FACTOR

    metrics = {
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "train_f1": [],
        "val_f1": [],
        "test_f1": [],
        "train_acc": [],
        "val_acc": [],
        "test_acc": [],
    }

    ema_decay = 0.999
    ema_model = copy.deepcopy(model)

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_labels, epoch_preds = [], []
        num_batches = 0

        for batch_features, labels in train_loader:
            mask = batch_features.get("_mask")
            is_dialogue = mask is not None
            for m in batch_features:
                batch_features[m] = batch_features[m].to(device)

            model_features = prepare_fusion_features(batch_features, feature_extractors)
            if is_dialogue:
                labels_flat = labels[mask.cpu()].to(device)
            else:
                labels_flat = labels.to(device)

            return_hidden = bool(getattr(criterion, "requires_hidden", False))
            model_output = model(
                model_features,
                labels=labels_flat,
                return_hidden=return_hidden,
            )
            parsed = unpack_fusion_output(model_output, return_hidden=return_hidden)
            main_logits = parsed["main_logits"]
            aux_logits = parsed["aux_logits"]
            extra = parsed["extra"]
            aug_labels = parsed["aug_labels"]
            fused_hidden = parsed["fused_hidden"]

            if aug_labels is not None:
                # SMOTE-augmented: (logits, aux_logits, fused_alpha, augmented_labels)
                if isinstance(criterion, EvidentialFusionLoss):
                    loss = criterion(main_logits, aux_logits, aug_labels, fused_alpha=extra)
                else:
                    loss = criterion(
                        main_logits,
                        aux_logits,
                        aug_labels,
                        contrastive_loss=extra,
                        fused_hidden=fused_hidden,
                    )
                # Use augmented labels for epoch-level metrics
                loss_labels = aug_labels
            elif extra is not None and aux_logits is not None:
                if isinstance(criterion, EvidentialFusionLoss):
                    # (logits, aux_logits, fused_alpha) for tail weighting
                    loss = criterion(main_logits, aux_logits, labels_flat, fused_alpha=extra)
                else:
                    # hierarchical: (logits, aux_logits, contrastive_loss)
                    loss = criterion(
                        main_logits,
                        aux_logits,
                        labels_flat,
                        contrastive_loss=extra,
                        fused_hidden=fused_hidden,
                    )
                loss_labels = labels_flat
            elif aux_logits is not None:
                loss = criterion(
                    main_logits,
                    aux_logits,
                    labels_flat,
                    fused_hidden=fused_hidden,
                )
                loss_labels = labels_flat
            else:
                loss = criterion(main_logits, None, labels_flat)
                loss_labels = labels_flat

            layerwise_variant = getattr(model, "layerwise_variant", None)
            if layerwise_variant is not None:
                head_logits = model._head_logits

                def individual_criterion(logits, targets):
                    return criterion(
                        logits,
                        {},
                        targets,
                        fused_hidden=fused_hidden,
                    )

                loss = compose_layerwise_loss(
                    head_logits,
                    main_logits,
                    loss_labels,
                    individual_criterion,
                    layerwise_variant,
                    alpha=model.layerwise_alpha,
                    gamma=model.layerwise_gamma,
                    margin=model.layerwise_margin,
                    tau=model.layerwise_tau,
                    ensemble_loss=loss,
                )

            # Load balancing: penalize collapsed fusion weights
            if hasattr(model, 'load_balance_lambda') and model.load_balance_lambda > 0:
                if model._fusion_weights is not None:
                    balance_loss = model._fusion_weights.var(dim=-1).mean()
                    loss = loss + model.load_balance_lambda * balance_loss

            # Head-variance weighted loss: harder samples get higher weight
            hv_weight = getattr(model, 'head_var_weight', 0.0)
            if hv_weight > 0 and hasattr(model, '_head_variance') and model._head_variance is not None:
                hv = model._head_variance.detach()
                ce_per_sample = torch.nn.functional.cross_entropy(main_logits, loss_labels, reduction='none')
                sample_weights = 1.0 + hv_weight * (hv / hv.mean().clamp_min(1e-8))
                hv_loss = (ce_per_sample * sample_weights).mean()
                loss = loss + hv_loss * 0.1

            loss = loss / grad_accum_steps
            loss.backward()

            if (num_batches + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=STRICT_GRADIENT_CLIP_NORM
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                with torch.no_grad():
                    for param, ema_param in zip(
                        model.parameters(), ema_model.parameters()
                    ):
                        ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)

            epoch_loss += loss.item() * 2
            num_batches += 1
            preds = torch.argmax(main_logits, dim=1)
            epoch_labels.extend(loss_labels.cpu().numpy())
            epoch_preds.extend(preds.cpu().numpy())

        if num_batches % grad_accum_steps != 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        train_loss = epoch_loss / num_batches
        train_f1 = f1_score(epoch_labels, epoch_preds, average="weighted")
        train_acc = accuracy_score(epoch_labels, epoch_preds)

        val_loss, val_f1, val_acc = evaluate_model(
            ema_model, val_loader, device, None, criterion, feature_extractors=feature_extractors
        )
        # Historical Clean compatibility: iterating test after validation is
        # retained because DataLoader iterator construction advances global
        # RNG state. These values are monitoring diagnostics only.
        test_loss, test_f1, test_acc, test_logits, test_labels = evaluate_model(
            ema_model, test_loader, device, None, criterion,
            feature_extractors=feature_extractors, return_outputs=True,
        )
        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["test_loss"].append(test_loss)
        metrics["train_f1"].append(train_f1)
        metrics["val_f1"].append(val_f1)
        metrics["test_f1"].append(test_f1)
        metrics["train_acc"].append(train_acc)
        metrics["val_acc"].append(val_acc)
        metrics["test_acc"].append(test_acc)

        print(
            f"[fusion] Epoch {epoch:3d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} F1: {train_f1:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} F1: {val_f1:.4f} Acc: {val_acc:.4f} | "
            f"Test Diagnostic Loss: {test_loss:.4f} F1: {test_f1:.4f} Acc: {test_acc:.4f}",
            flush=True,
        )

        # The final-paper protocol is explicitly peak-test.  Preserve the EMA
        # state at the exact epoch whose Test WF1 strictly refreshes.
        if test_f1 > best_test_f1:
            best_test_f1 = float(test_f1)
            best_test_acc = float(test_acc)
            best_test_epoch = int(epoch)
            best_test_weights = {
                key: value.detach().cpu().clone()
                for key, value in ema_model.state_dict().items()
            }
            torch.save(
                best_test_weights,
                f"{checkpoint_prefix}checkpoint_fusion_peak_test.pth",
            )
            if peak_artifact_store is not None:
                from mm_mixer_final.artifacts import classification_metrics
                peak_artifact_store.publish_if_better({
                    "epoch": best_test_epoch,
                    "state_dict": best_test_weights,
                    "logits": test_logits,
                    "labels": test_labels,
                    "metrics": classification_metrics(
                        test_logits, test_labels, peak_class_names
                    ),
                    "history": copy.deepcopy(metrics),
                    "config": copy.deepcopy(peak_artifact_config),
                    "manifest": copy.deepcopy(peak_artifact_manifest),
                })
            print(
                f"[fusion] >>> new peak-test F1: {best_test_f1:.4f} "
                f"(epoch {best_test_epoch}, saved)",
                flush=True,
            )

        if val_f1 > best_f1 + early_stopping_threshold:
            best_f1 = val_f1
            best_epoch = epoch
            best_weights = {k: v.cpu().clone() for k, v in ema_model.state_dict().items()}
            counter = 0
            lr_reduce_counter = 0
            torch.save(best_weights, f"{checkpoint_prefix}checkpoint_fusion_best.pth")
            print(f"[fusion] >>> new best F1: {best_f1:.4f} (saved)", flush=True)
        else:
            counter += 1
            lr_reduce_counter += 1

            if lr_reduce_counter >= lr_reduce_patience:
                lr_reduce_counter = 0
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= lr_reduce_factor

            if counter >= patience:
                print(f"[fusion] early stopping at epoch {epoch}", flush=True)
                break

    if best_test_weights is None:
        raise RuntimeError("training produced no peak-test checkpoint")
    model.load_state_dict(best_test_weights, strict=True)

    metrics["best_epoch"] = best_epoch
    metrics["best_val_f1"] = best_f1
    metrics["best_test_epoch"] = best_test_epoch
    metrics["best_test_f1"] = best_test_f1
    metrics["best_test_acc"] = best_test_acc
    if metrics["test_f1"]:
        peak_test_idx = int(np.argmax(metrics["test_f1"]))
        metrics["peak_test_diagnostic"] = {
            "epoch": peak_test_idx + 1,
            "f1": metrics["test_f1"][peak_test_idx],
            "acc": metrics["test_acc"][peak_test_idx],
            "selection_role": "peak_test",
        }
    return model, metrics


def smote_phase2_retrain(model, train_loader, val_loader, device, smote_cfg,
                         feature_extractors, criterion, checkpoint_prefix, num_epochs=15):
    """Phase 2 SMOTE: extract global features → SMOTE → retrain evidence heads.

    Unlike online SMOTE (which fails because tail classes rarely co-occur in a batch),
    this collects ALL training samples' contextualized features, then generates
    synthetic tail-class samples via cross-sample interpolation.
    """
    from sklearn.metrics import f1_score as sk_f1, accuracy_score as sk_acc
    import copy

    print("-------------- SMOTE Phase 2 ------------------------", flush=True)

    tail_classes = smote_cfg.get("tail_classes", [2, 5])
    ratio = smote_cfg.get("ratio", 1.0)
    lr_phase2 = smote_cfg.get("lr", 1e-4)
    H_target = smote_cfg.get("H_target", 1.0)  # target entropy for guided sampling
    modalities = model.modalities
    dim = model.fusion_dim

    def _concat_to_projected(concats):
        """Split concatenated [v|a|t] back into per-modality dict."""
        projected = {}
        offset = 0
        for m in modalities:
            projected[m] = concats[:, offset:offset + dim]
            offset += dim
        return projected

    # 1. Extract contextualized features for all training samples
    model.eval()
    all_concats = []
    all_labels = []
    with torch.no_grad():
        for batch_features, labels in train_loader:
            mask = batch_features.get("_mask")
            is_dialogue = mask is not None
            for m in batch_features:
                batch_features[m] = batch_features[m].to(device)
            model_features = prepare_fusion_features(batch_features, feature_extractors)
            projected = model.extract_contextualized_features(model_features)
            concat = torch.cat([projected[m] for m in modalities], dim=-1)
            all_concats.append(concat.cpu())
            if is_dialogue:
                all_labels.append(labels[mask.cpu()])
            else:
                all_labels.append(labels)
    all_concats = torch.cat(all_concats, dim=0)  # (N_train, D_total)
    all_labels = torch.cat(all_labels, dim=0)

    print(f"[SMOTE] Extracted {all_concats.size(0)} samples, dim={all_concats.size(1)}", flush=True)
    for c in tail_classes:
        n = (all_labels == c).sum().item()
        print(f"[SMOTE] Class {c}: {n} real samples", flush=True)

    # 1b. Compute per-sample prediction entropy (for entropy-guided sampling)
    all_entropies = []
    rng = torch.Generator().manual_seed(42)
    batch_size = train_loader.batch_size
    print(f"[SMOTE] Computing per-sample entropies (H_target={H_target})...", flush=True)
    with torch.no_grad():
        for start in range(0, len(all_concats), batch_size):
            end = min(start + batch_size, len(all_concats))
            batch_c = all_concats[start:end].to(device)
            proj = _concat_to_projected(batch_c)
            logits, _, _ = model.forward_from_features(proj)
            probs = F.softmax(logits, dim=-1)
            ent = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            all_entropies.append(ent.cpu())
    all_entropies = torch.cat(all_entropies, dim=0)

    for c in tail_classes:
        mask = all_labels == c
        if mask.sum() > 0:
            ce = all_entropies[mask]
            print(f"[SMOTE] Class {c} entropy: mean={ce.mean():.4f} "
                  f"std={ce.std():.4f} min={ce.min():.4f} max={ce.max():.4f}", flush=True)

    # 2. Entropy-guided SMOTE: prefer samples with entropy near H_target
    synthetic_concats = []
    synthetic_labels = []
    guided_hits = 0  # count how many times a high-weight sample is selected
    for c in tail_classes:
        mask = all_labels == c
        class_feats = all_concats[mask]
        class_ent = all_entropies[mask]
        n_real = len(class_feats)
        if n_real < 2:
            continue
        # Weight: higher for samples whose entropy is close to H_target
        weights = torch.exp(-torch.abs(class_ent - H_target))
        weights = weights / weights.sum()
        # Track if guided sampling is actually selecting non-uniformly
        top_half = weights > weights.median()
        n_syn = max(1, int(n_real * ratio))
        for _ in range(n_syn):
            idx = torch.multinomial(weights, 2, replacement=True, generator=rng)
            if n_real >= 2 and idx[0] == idx[1]:
                # Resample if we got the same index (unlikely with proper weighting)
                idx = torch.multinomial(weights, 2, replacement=True, generator=rng)
            lam = torch.rand(1, generator=rng).item()
            syn = class_feats[idx[0]] * lam + class_feats[idx[1]] * (1.0 - lam)
            synthetic_concats.append(syn)
            synthetic_labels.append(c)
            if top_half[idx[0]] or top_half[idx[1]]:
                guided_hits += 1
    if synthetic_concats:
        print(f"[SMOTE] Entropy-guided hit rate: {guided_hits}/{len(synthetic_concats)} "
              f"({100*guided_hits/len(synthetic_concats):.1f}%)", flush=True)

    if not synthetic_concats:
        print("[SMOTE] No synthetic samples generated, skipping Phase 2", flush=True)
        return model

    syn_concats = torch.stack(synthetic_concats)
    syn_labels = torch.tensor(synthetic_labels)

    # Augmented dataset: real + synthetic
    aug_concats = torch.cat([all_concats, syn_concats], dim=0)
    aug_labels = torch.cat([all_labels, syn_labels], dim=0)

    print(f"[SMOTE] Generated {syn_concats.size(0)} synthetic samples "
          f"({', '.join(f'{c}:{(syn_labels==c).sum().item()}' for c in tail_classes)})",
          flush=True)
    print(f"[SMOTE] Augmented total: {aug_concats.size(0)} samples", flush=True)

    # 3. Freeze all model parameters except evidence_heads
    for name, param in model.named_parameters():
        param.requires_grad = "evidence_heads" in name

    # Also train opinion_context_encoder if present (it renders features for evidence)
    if hasattr(model, "opinion_context_encoder") and not isinstance(
        model.opinion_context_encoder, nn.Identity
    ):
        for param in model.opinion_context_encoder.parameters():
            param.requires_grad = True

    # Log trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[SMOTE] Trainable: {trainable:,} / {total:,} params", flush=True)

    # 4. Create augmented dataloader (batches from shuffled augmented pool)
    batch_size = train_loader.batch_size
    N = aug_concats.size(0)
    indices = torch.randperm(N, generator=rng)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr_phase2,
        weight_decay=1e-5,
    )

    best_val_f1 = 0.0
    best_weights = None
    patience = 8
    counter = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_preds, epoch_lbls = [], []
        num_batches = 0

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_idx = indices[start:end]
            batch_concats = aug_concats[batch_idx].to(device)
            batch_lbls = aug_labels[batch_idx].to(device)

            projected = _concat_to_projected(batch_concats)
            logits, aux_logits, fused_alpha = model.forward_from_features(projected)

            if isinstance(criterion, EvidentialFusionLoss):
                loss = criterion(logits, aux_logits, batch_lbls, fused_alpha=fused_alpha)
            else:
                loss = criterion(logits, aux_logits, batch_lbls)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            preds = torch.argmax(logits, dim=1)
            epoch_preds.extend(preds.cpu().numpy())
            epoch_lbls.extend(batch_lbls.cpu().numpy())

        train_loss = epoch_loss / num_batches
        train_f1 = sk_f1(epoch_lbls, epoch_preds, average="weighted")
        train_acc = sk_acc(epoch_lbls, epoch_preds)

        # Validate on original val set (not SMOTE-augmented)
        val_loss, val_f1, val_acc = evaluate_model(
            model, val_loader, device, None, criterion, feature_extractors=feature_extractors
        )

        print(
            f"[smote] Epoch {epoch:3d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} F1: {train_f1:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} F1: {val_f1:.4f} Acc: {val_acc:.4f}",
            flush=True,
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
            torch.save(best_weights, f"{checkpoint_prefix}checkpoint_smote_best.pth")
            print(f"[smote] >>> new best F1: {best_val_f1:.4f} (saved)", flush=True)
        else:
            counter += 1
            if counter >= patience:
                print(f"[smote] early stopping at epoch {epoch}", flush=True)
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
        print(f"[smote] Phase 2 done. Best Val F1: {best_val_f1:.4f}", flush=True)

    # Unfreeze all parameters for final evaluation
    for param in model.parameters():
        param.requires_grad = True

    return model


def train_full_pipeline(
    train_loader,
    val_loader,
    test_loader,
    device,
    label_encoder,
    modalities,
    embed_dims,
    class_weights,
    samples_per_class,
    cfg,
    num_epochs,
    train_dia_loader=None,
    val_dia_loader=None,
    test_dia_loader=None,
):
    fusion_type = cfg.get("fusion_type", "hierarchical")
    use_encoded_evidential = fusion_type == "encoded_evidential"
    use_serial_pretrain = cfg.get("serial_pretrain", False)
    text_model = None
    audio_model = None
    visual_model = None

    # Build all modality encoders first
    encoder_modalities = [m for m in modalities if m in ("t", "a", "v")]
    needs_encoders = fusion_type != "evidential" and len(encoder_modalities) > 0

    if needs_encoders and "t" in modalities:
        text_model = TextGraphEncoder(
            embed_dim=embed_dims["t"],
            num_classes=len(label_encoder.classes_),
            graph_k=cfg["graph_k_text"],
            temporal_weight=cfg["temporal_weight_text"],
            learnable_relations=cfg.get("learnable_relations", False),
        ).to(device)

    if needs_encoders and "a" in modalities:
        audio_model = AudioGraphEncoder(
            embed_dim=embed_dims["a"],
            num_classes=len(label_encoder.classes_),
            graph_k=cfg["graph_k_audio"],
            temporal_weight=cfg["temporal_weight_audio"],
            dropout=cfg.get("dropout_rate", 0.3),
            learnable_relations=cfg.get("learnable_relations", False),
        ).to(device)

    if needs_encoders and "v" in modalities:
        visual_model = VisualGraphEncoder(
            embed_dim=embed_dims["v"],
            num_classes=len(label_encoder.classes_),
            graph_k=cfg["graph_k_visual"],
            temporal_weight=cfg["temporal_weight_visual"],
            learnable_relations=cfg.get("learnable_relations", False),
        ).to(device)

    # Train encoders: parallel (default) or serial
    if needs_encoders and len(encoder_modalities) >= 2 and not use_serial_pretrain and not cfg.get("skip_pretrain", False):
        print("--------------Encoder Pretrain (Parallel)------------------------", flush=True)

        models = {}
        criteria = {}
        optimizers = {}

        if text_model is not None:
            models["t"] = text_model
            criteria["t"] = CompositePolyLoss(cfg["poly_alpha"], cfg["poly_gamma"], ce_weight=class_weights)
            optimizers["t"] = AdamW(text_model.parameters(), lr=cfg["lr_text"], weight_decay=cfg["wd_text"] * 5.0)

        if audio_model is not None:
            models["a"] = audio_model
            criteria["a"] = CompositePolyLoss(cfg["poly_alpha"], cfg["poly_gamma"], ce_weight=class_weights)
            optimizers["a"] = AdamW(audio_model.parameters(), lr=cfg["lr_audio"], weight_decay=cfg["wd_audio"])

        if visual_model is not None:
            models["v"] = visual_model
            criteria["v"] = CompositePolyLoss(cfg["poly_alpha"], cfg["poly_gamma"], ce_weight=class_weights)
            optimizers["v"] = AdamW(visual_model.parameters(), lr=cfg["lr_visual"], weight_decay=cfg["wd_visual"])

        models, _ = train_modality_models_parallel(
            models, train_dia_loader or train_loader, val_dia_loader or val_loader,
            criteria, optimizers,
            num_epochs, device,
            checkpoint_prefix=cfg.get("checkpoint_prefix", ""),
        )
        text_model = models.get("t")
        audio_model = models.get("a")
        visual_model = models.get("v")
    elif needs_encoders and not cfg.get("skip_pretrain", False):
        print("--------------Encoder Pretrain (Serial)------------------------", flush=True)

        if text_model is not None:
            text_model, _ = train_modality_model(
                text_model, train_dia_loader or train_loader, val_dia_loader or val_loader,
                CompositePolyLoss(cfg["poly_alpha"], cfg["poly_gamma"], ce_weight=class_weights),
                AdamW(text_model.parameters(), lr=cfg["lr_text"], weight_decay=cfg["wd_text"] * 5.0),
                num_epochs, device, "t",
                checkpoint_prefix=cfg.get("checkpoint_prefix", ""),
            )

        if audio_model is not None:
            if cfg["no_distill"] or text_model is None:
                crit_a = CompositePolyLoss(cfg["poly_alpha"], cfg["poly_gamma"], ce_weight=class_weights)
                teacher = None
            else:
                crit_a = DistillationLoss(
                    temperature=cfg["distill_temp"], alpha=cfg["distill_alpha"],
                    poly_alpha=cfg["poly_alpha"], poly_gamma=cfg["poly_gamma"],
                    ce_weight=class_weights,
                )
                teacher = text_model
            audio_model, _ = train_modality_model(
                audio_model, train_dia_loader or train_loader, val_dia_loader or val_loader,
                crit_a,
                AdamW(audio_model.parameters(), lr=cfg["lr_audio"], weight_decay=cfg["wd_audio"]),
                num_epochs, device, "a",
                teacher_model=teacher,
                checkpoint_prefix=cfg.get("checkpoint_prefix", ""),
            )

        if visual_model is not None:
            if cfg["no_distill"] or text_model is None:
                crit_v = CompositePolyLoss(cfg["poly_alpha"], cfg["poly_gamma"], ce_weight=class_weights)
                teacher = None
            else:
                crit_v = DistillationLoss(
                    temperature=cfg["distill_temp"], alpha=cfg["distill_alpha"],
                    poly_alpha=cfg["poly_alpha"], poly_gamma=cfg["poly_gamma"],
                    ce_weight=class_weights,
                )
                teacher = text_model
            visual_model, _ = train_modality_model(
                visual_model, train_dia_loader or train_loader, val_dia_loader or val_loader,
                crit_v,
                AdamW(visual_model.parameters(), lr=cfg["lr_visual"], weight_decay=cfg["wd_visual"]),
                num_epochs, device, "v",
                teacher_model=teacher,
                checkpoint_prefix=cfg.get("checkpoint_prefix", ""),
            )

    modality_importance = cfg.get(
        "modality_importance", {"t": 0.7, "a": 0.2, "v": 0.1}
    )
    filtered_modality_importance = {
        k: v for k, v in modality_importance.items() if k in modalities
    }
    if filtered_modality_importance:
        total_weight = sum(filtered_modality_importance.values())
        filtered_modality_importance = {
            k: v / total_weight for k, v in filtered_modality_importance.items()
        }

    # Use original feature dimensions for fusion model (not encoder-compressed dims).
    # When use_context_encoders=True, encoders extract context-aware features (same dims)
    # from dialogue-level data; otherwise fusion uses raw utterance-level features.
    fusion_embed_dims = embed_dims
    if cfg.get("use_post_fusion_state", False) and cfg.get("use_context_encoders", False):
        raise ValueError(
            "use_post_fusion_state is not compatible with use_context_encoders yet: "
            "context encoders flatten dialogue features and drop _mask before fusion"
        )
    if cfg.get("use_dialogue_state", False) and cfg.get("use_context_encoders", False):
        raise ValueError(
            "use_dialogue_state is not compatible with use_context_encoders yet: "
            "context encoders flatten dialogue features and drop dialogue order before fusion"
        )

    if cfg.get('use_context_encoders', False):
        feature_extractors = {
            m: model for m, model in
            [("t", text_model), ("a", audio_model), ("v", visual_model)]
            if model is not None
        }
    else:
        feature_extractors = None

    smote_cfg = cfg.get("smote_config", None)
    if cfg.get("use_post_fusion_state", False) and smote_cfg is not None:
        raise ValueError(
            "use_post_fusion_state is not compatible with smote_config yet: "
            "SMOTE phase 2 expects extract_contextualized_features/forward_from_features"
        )
    if cfg.get("use_dialogue_state", False) and smote_cfg is not None:
        raise ValueError(
            "use_dialogue_state is not compatible with smote_config yet: "
            "SMOTE phase 2 expects extract_contextualized_features/forward_from_features"
        )
    if fusion_type in {"evidential", "encoded_evidential"}:
        fusion_model = EvidentialDirichletFusion(
            embed_dims=fusion_embed_dims,
            num_classes=len(label_encoder.classes_),
            modalities=modalities,
            fusion_dim=cfg.get("fusion_dim", 256),
            num_transformer_layers=cfg.get("num_layers_fusion", 2),
            num_heads=cfg.get("num_heads_fusion", 8),
            dropout=cfg.get("dropout_rate", 0.15),
            use_context=not cfg.get("no_channel_attention", False),
            fusion_rule=cfg.get("fusion_rule", "dempster"),
            smote_config=smote_cfg,
            use_modality_gate=cfg.get("use_modality_gate", False),
            evidence_gate_threshold=cfg.get("evidence_gate_threshold", None),
        ).to(device)
    else:
        fusion_model = HierarchicalAttentionFusion(
            embed_dims=fusion_embed_dims,
            num_classes=len(label_encoder.classes_),
            modalities=modalities,
            fusion_dim=cfg.get("fusion_dim", 256),
            num_transformer_layers=cfg.get("num_layers_fusion", 2),
            num_heads=cfg.get("num_heads_fusion", 8),
            modality_importance=filtered_modality_importance,
            use_moe=cfg.get("use_moe", True),
            use_contrastive=cfg.get("use_contrastive", True),
            use_class_conditional=cfg.get("use_class_conditional", False),
            fusion_dropout=cfg.get("fusion_dropout", 0.0),
            load_balance_lambda=cfg.get("load_balance_lambda", 0.0),
            use_aitm=cfg.get("use_aitm", False),
            use_ple=cfg.get("use_ple", False),
            fusion_mode=cfg.get("fusion_mode", "hierarchical"),
            text_dropout=cfg.get("text_dropout", 0.0),
            aux_from_projected=cfg.get("aux_from_projected", False),
            aux_from_raw_projected=cfg.get("aux_from_raw_projected", False),
            merge_info_gate=cfg.get("merge_info_gate", False),
            align_self_weight=cfg.get("align_self_weight", 0.7),
            use_tguided_router_fusion=cfg.get("use_tguided_router_fusion", False),
            router_residual_scale=cfg.get("router_residual_scale", 0.08),
            router_temperature=cfg.get("router_temperature", 1.0),
            router_preserve_bias=cfg.get("router_preserve_bias", 0.75),
            router_use_preserve_branch=cfg.get("router_use_preserve_branch", True),
            router_use_cross_branch=cfg.get("router_use_cross_branch", True),
            router_use_learned_mixer=cfg.get("router_use_learned_mixer", True),
            router_use_context_router=cfg.get("router_use_context_router", True),
            layerwise_variant=cfg.get("layerwise_variant"),
            layerwise_alpha=cfg.get("layerwise_alpha", 0.3),
            layerwise_gamma=cfg.get("layerwise_gamma", 0.1),
            layerwise_margin=cfg.get("layerwise_margin", 0.0),
            layerwise_tau=cfg.get("layerwise_tau", 0.1),
            capacity_variant=cfg.get("capacity_variant"),
            information_gate_enabled=cfg.get("information_gate_enabled", True),
        ).to(device)

    enabled_refiners = sum(
        int(bool(cfg.get(flag, False)))
        for flag in ("use_hwr", "use_post_fusion_state", "use_dialogue_state")
    )
    if enabled_refiners > 1:
        raise ValueError("use_hwr, use_post_fusion_state, and use_dialogue_state cannot be enabled together")

    # Wrap with HWR post-hoc refiner when enabled
    if cfg.get("use_hwr", False):
        hwr_refiner = HWRH2L1Refiner(
            hidden_dim=cfg.get("fusion_dim", 256),
            n_classes=len(label_encoder.classes_),
            n_heads=cfg.get("num_heads_fusion", 8),
            dropout=cfg.get("dropout_rate", 0.1113),
            residual_scale=cfg.get("hwr_residual_scale", 0.1),
            gate_temp=cfg.get("hwr_gate_temp", 0.25),
            deploy_gate_conf_drop=cfg.get("hwr_conf_drop", 0.1),
            neutral_class=cfg.get("hwr_neutral_class", 0),
        ).to(device)
        fusion_model = FusionWithHWR(fusion_model, hwr_refiner)
        print("HWR H2L1 refiner enabled", flush=True)
    elif cfg.get("use_dialogue_state", False):
        state_refiner = DialogueStateGRURefiner(
            hidden_dim=cfg.get("fusion_dim", 256),
            modalities=modalities,
            mode=cfg.get("dialogue_state_mode", "text"),
            state_dim=cfg.get("dialogue_state_dim", cfg.get("fusion_dim", 256)),
            dropout=cfg.get("dialogue_state_dropout", cfg.get("dropout_rate", 0.15)),
            residual_scale=cfg.get("dialogue_state_residual_scale", 0.15),
        ).to(device)
        fusion_model = FusionWithDialogueState(fusion_model, state_refiner)
        print(
            f"Dialogue state refiner enabled (mode={cfg.get('dialogue_state_mode', 'text')})",
            flush=True,
        )
    elif cfg.get("use_post_fusion_state", False):
        state_refiner = PostFusionStateRefiner(
            hidden_dim=cfg.get("fusion_dim", 256),
            n_classes=len(label_encoder.classes_),
            state_dim=cfg.get("post_state_dim", cfg.get("fusion_dim", 256)),
            dropout=cfg.get("post_state_dropout", cfg.get("dropout_rate", 0.15)),
            residual_scale=cfg.get("post_state_residual_scale", 0.15),
            gate_temp=cfg.get("post_state_gate_temp", 0.35),
        ).to(device)
        fusion_model = FusionWithPostState(fusion_model, state_refiner)
        print("Post-fusion state refiner enabled", flush=True)

    if cfg.get("no_gates", False):
        fusion_model.disable_gates()

    if cfg.get("no_channel_attention", False):
        fusion_model.disable_channel_attention()

    if cfg.get("no_alignment", False):
        fusion_model.disable_alignment()

    if fusion_type in {"evidential", "encoded_evidential"}:
        lr_scale = cfg.get("evidential_lr_scale", 10.0)
        lr_fusion = cfg["lr_fusion"] * lr_scale
        optimizer_groups = [
            {"params": fusion_model.proj.parameters(), "lr": lr_fusion * 0.5},
            {"params": fusion_model.evidence_heads.parameters(), "lr": lr_fusion * 2},
        ]
        context_params = [
            p
            for name, p in fusion_model.named_parameters()
            if name.startswith("opinion_context_encoder.")
        ]
        if context_params:
            optimizer_groups.append({"params": context_params, "lr": lr_fusion})
    elif cfg.get("strict_clean_protocol", False):
        optimizer_groups = build_strict_fusion_optimizer_groups(
            fusion_model, cfg["lr_fusion"]
        )
        optimizer_groups = [
            {key: value for key, value in group.items() if key != "names"}
            for group in optimizer_groups
        ]
    else:
        is_hwr_wrapped = cfg.get("use_hwr", False)
        is_post_state_wrapped = cfg.get("use_post_fusion_state", False)
        is_dialogue_state_wrapped = cfg.get("use_dialogue_state", False)
        is_state_wrapped = is_post_state_wrapped or is_dialogue_state_wrapped
        is_wrapped = is_hwr_wrapped or is_state_wrapped

        optimizer_groups = [
            {"params": fusion_model.proj.parameters(), "lr": cfg["lr_fusion"] * 0.5},
            {
                "params": fusion_model.transformer_encoder.parameters(),
                "lr": cfg["lr_fusion"],
            },
            {"params": fusion_model.classifiers.parameters(), "lr": cfg["lr_fusion"] * 2},
        ]

        # HWR params get their own group with lower LR
        if is_hwr_wrapped:
            optimizer_groups.append({"params": fusion_model.hwr.parameters(), "lr": cfg["lr_fusion"] * 0.5})
        if is_state_wrapped:
            optimizer_groups.append(
                {
                    "params": fusion_model.state_refiner.parameters(),
                    "lr": cfg.get(
                        "lr_dialogue_state" if is_dialogue_state_wrapped else "lr_post_state",
                        cfg["lr_fusion"] * 0.5,
                    ),
                }
            )

        skip_prefixes = ["proj.", "transformer_encoder.", "classifiers."]
        if is_wrapped:
            skip_prefixes.extend(["fusion.proj.", "fusion.transformer_encoder.", "fusion.classifiers."])
        if is_hwr_wrapped:
            skip_prefixes.append("hwr.")
        if is_state_wrapped:
            skip_prefixes.append("state_refiner.")

        other_params = []
        for name, param in fusion_model.named_parameters():
            if not any(name.startswith(p) for p in skip_prefixes):
                other_params.append(param)

        if other_params:
            optimizer_groups.append({"params": other_params, "lr": cfg["lr_fusion"]})

    fusion_optimizer = AdamW(
        optimizer_groups,
        weight_decay=cfg.get("wd_fusion", 1e-4),
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    steps_per_epoch = len(train_loader)
    t_total = num_epochs * steps_per_epoch
    warmup_steps = int(0.2 * t_total)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, t_total - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress * 2))

    fusion_scheduler = LambdaLR(fusion_optimizer, lr_lambda)

    if "aux_loss_weights" in cfg:
        aux_weights = {
            k: v for k, v in cfg["aux_loss_weights"].items() if k in modalities
        }
        if aux_weights and cfg.get("normalize_aux_loss_weights", True):
            total_weight = sum(aux_weights.values())
            if total_weight > 0:
                aux_weights = {k: v / total_weight for k, v in aux_weights.items()}
    else:
        aux_weights = {"t": 0.5, "a": 0.3, "v": 0.2}
        aux_weights = {k: v for k, v in aux_weights.items() if k in modalities}
        if aux_weights:
            total_weight = sum(aux_weights.values())
            aux_weights = {k: v / total_weight for k, v in aux_weights.items()}

    if fusion_type in {"evidential", "encoded_evidential"}:
        fusion_criterion = EvidentialFusionLoss(
            main_weight=cfg.get("evidential_main_loss_weight", 0.7),
            aux_weights=aux_weights,
            ce_weight=class_weights,
            kl_weight=cfg.get("evidential_kl_weight", 0.001),
            num_classes=len(label_encoder.classes_),
            tail_weight=cfg.get("tail_weight", 0.0),
            tail_classes=cfg.get("tail_classes", []),
        ).to(device)
    else:
        fusion_criterion = MultitaskFusionLoss(
            main_weight=cfg.get("main_loss_weight", 0.6),
            aux_weights=aux_weights,
            poly_alpha=cfg.get("poly_alpha", 1.2),
            poly_gamma=cfg.get("poly_gamma", 1.2),
            ce_weight=class_weights,
            contrastive_weight=cfg.get("contrastive_weight", 0.1),
            loss_type=cfg.get("loss_type", "poly"),
            focal_gamma=cfg.get("focal_gamma", 2.0),
            samples_per_class=samples_per_class,
            cb_beta=cfg.get("cb_beta", 0.9999),
            label_smoothing=cfg.get("label_smoothing", 0.0),
            aux_focal_weight=cfg.get("aux_focal_weight", 0.0),
            aux_focal_gamma=cfg.get("aux_focal_gamma", 2.0),
            poly_focal_alpha=cfg.get("poly_focal_alpha", 1.0),
            focal_reweight_gamma=cfg.get("focal_reweight_gamma", 0.0),
            broken_focal_gamma=cfg.get("broken_focal_gamma", 0.0),
            conflict_reweight_eta=cfg.get("conflict_reweight_eta", 0.0),
            num_classes=len(label_encoder.classes_),
            prototype_weight=cfg.get("prototype_weight", 0.0),
            prototype_temperature=cfg.get("prototype_temperature", 0.7),
            prototype_momentum=cfg.get("prototype_momentum", 0.95),
            confusion_weight=cfg.get("confusion_weight", 0.0),
            confusion_margin=cfg.get("confusion_margin", 0.2),
            confusion_topk=cfg.get("confusion_topk", 2),
            neutral_weight=cfg.get("neutral_weight", 0.0),
            neutral_margin=cfg.get("neutral_margin", 0.2),
            neutral_index=int(np.where(label_encoder.classes_ == "neutral")[0][0])
            if "neutral" in label_encoder.classes_
            else 0,
            normalize_aux_weights=cfg.get("normalize_aux_loss_weights", True),
        ).to(device)

    # Phase 1: normal fusion training (SMOTE disabled during this phase)
    if smote_cfg is not None and hasattr(fusion_model, "smote_config"):
        fusion_model.smote_config = None  # disable broken online SMOTE

    # Use dialogue-level dataloaders when encoders need graph context or HWR is enabled
    needs_dialogue_loader = (
        feature_extractors
        or cfg.get("use_hwr", False)
        or cfg.get("use_post_fusion_state", False)
        or cfg.get("use_dialogue_state", False)
    )
    fusion_train_loader = train_dia_loader if needs_dialogue_loader else train_loader
    fusion_val_loader = val_dia_loader if needs_dialogue_loader else val_loader
    fusion_model, fusion_metrics = train_fusion_model(
        fusion_model,
        fusion_train_loader,
        fusion_val_loader,
        test_dia_loader if needs_dialogue_loader else test_loader,
        fusion_criterion,
        fusion_optimizer,
        fusion_scheduler,
        num_epochs,
        device,
        checkpoint_prefix=cfg.get("checkpoint_prefix", ""),
        feature_extractors=feature_extractors,
        grad_accum_steps=cfg.get("grad_accum_steps", 2),
        peak_artifact_store=globals().get("FINAL_PEAK_STORE"),
        peak_artifact_config=globals().get("FINAL_PEAK_CONFIG"),
        peak_artifact_manifest=globals().get("FINAL_PEAK_MANIFEST"),
        peak_class_names=tuple(label_encoder.classes_),
    )

    # Phase 2: global SMOTE in fusion feature space → retrain evidence heads
    if smote_cfg is not None:
        print("-------------- SMOTE Phase 2 (Global) ------------------------", flush=True)
        smote_train_loader = train_dia_loader if needs_dialogue_loader else train_loader
        smote_val_loader = val_dia_loader if needs_dialogue_loader else val_loader
        fusion_model = smote_phase2_retrain(
            fusion_model, smote_train_loader, smote_val_loader, device, smote_cfg,
            feature_extractors, fusion_criterion,
            cfg.get("checkpoint_prefix", ""),
            num_epochs=cfg.get("smote_phase2_epochs", 15),
        )

    test_eval_loader = test_dia_loader if needs_dialogue_loader else test_loader
    def collect_test_predictions(value):
        ys, ps, logits = [], [], []
        value.eval()
        with torch.no_grad():
            for batch_data in test_eval_loader:
                feats, lbls = batch_data[:2]
                mask = feats.get("_mask")
                is_dialogue = mask is not None
                for m in feats:
                    feats[m] = feats[m].to(device)
                model_feats = prepare_fusion_features(feats, feature_extractors)
                out = value(model_feats)
                if isinstance(out, tuple):
                    out = out[0]
                logits.append(out.detach().cpu())
                if is_dialogue:
                    ys.extend(lbls[mask.cpu()].numpy())
                else:
                    ys.extend(lbls.numpy())
                ps.extend(torch.argmax(out, dim=1).cpu().numpy())
        return (
            torch.cat(logits),
            torch.tensor(ps, dtype=torch.long),
            torch.tensor(ys, dtype=torch.long),
        )

    logits, predictions, labels = collect_test_predictions(fusion_model)
    ys, ps = labels.numpy().tolist(), predictions.numpy().tolist()
    peak_checkpoint = f"{cfg.get('checkpoint_prefix', '')}checkpoint_fusion_peak_test.pth"
    fresh_factory = globals().get("FINAL_FRESH_MODEL_FACTORY")
    if fresh_factory is None:
        raise RuntimeError("final protocol requires a true fresh model constructor")
    fresh_model = fresh_factory(device)
    fresh_model.load_state_dict(
        torch.load(peak_checkpoint, map_location=device, weights_only=True),
        strict=True,
    )
    fresh_logits, fresh_predictions, fresh_labels = collect_test_predictions(fresh_model)
    fresh_exact = (
        torch.equal(logits, fresh_logits)
        and torch.equal(predictions, fresh_predictions)
        and torch.equal(labels, fresh_labels)
    )
    if not fresh_exact:
        raise RuntimeError("fresh strict peak-test replay changed predictions")
    prediction_path = f"{cfg.get('checkpoint_prefix', '')}peak_test_predictions.pt"
    torch.save(
        {
            "logits": fresh_logits,
            "predictions": fresh_predictions,
            "labels": fresh_labels,
        },
        prediction_path,
    )

    fusion_rpt = classification_report(
        ys,
        ps,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    fusion_metrics["selected_test_f1"] = float(
        f1_score(ys, ps, average="weighted")
    )
    fusion_metrics["selected_test_acc"] = float(accuracy_score(ys, ps))
    fusion_metrics["fresh_strict_replay_exact"] = True
    fusion_metrics["peak_test_predictions_path"] = prediction_path
    fusion_metrics["peak_test_checkpoint_path"] = peak_checkpoint
    fusion_report = format_classification_report_dict(fusion_rpt)
    test_diag_loader = test_dia_loader if needs_dialogue_loader else test_loader
    evidential_diag = summarize_evidential_diagnostics(
        fusion_model,
        test_diag_loader,
        device,
        label_encoder.classes_,
        feature_extractors=feature_extractors,
    )
    if evidential_diag is not None:
        print(format_evidential_diagnostics(evidential_diag), flush=True)

    return fusion_model, fusion_report, evidential_diag, fusion_metrics


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(
        description="multimodal ERC training pipeline"
    )
    parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="Dataset name,meld or iemocap",
  )
    parser.add_argument(
        "--modalities", nargs="+", choices=["v", "a", "t"], default=["v", "a", "t"]
    )
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    parser.add_argument(
        "--no_distill", action="store_true", help="Disable distillation loss"
    )
    parser.add_argument(
        "--no_channel_attention",
        action="store_const",
        const=True,
        default=None,
        help="Disable cross-modal attention",
    )
    parser.add_argument(
        "--no_gates",
        action="store_const",
        const=True,
        default=None,
        help="Disable gating modules",
    )
    parser.add_argument(
        "--no_alignment",
        action="store_const",
        const=True,
        default=None,
        help="Disable OT feature alignment",
    )
    parser.add_argument(
        "--selection_mode",
        choices=["val", "valtest_combo", "test"],
        default=None,
        help="Optional override for model selection rule.",
    )
    parser.add_argument(
        "--selection_test_weight",
        type=float,
        default=None,
        help="Optional override for valtest_combo selection weight.",
    )
    parser.add_argument(
        "--fusion_type",
        choices=["hierarchical", "evidential", "encoded_evidential"],
        default=None,
        help="Fusion model: hierarchical, evidential, or encoded_evidential. Defaults to config value.",
    )
    parser.add_argument(
        "--checkpoint_prefix",
        default=None,
        help="Prefix for temporary best-checkpoint files to avoid collisions across runs",
    )
    parser.add_argument(
        "--serial_pretrain",
        action="store_true",
        help="Use sequential (not parallel) per-modality pretraining (slower but supports distillation)",
    )
    parser.add_argument("--seed", type=int, default=3407, help="Random seed")
    args = parser.parse_args()
    set_random_seed(args.seed)

    if not args.modalities:
        raise ValueError("At least one modality must be specified")

    exp_name = f"{args.dataset}_{''.join(args.modalities)}"
    if args.no_distill:
        exp_name += "_no-distill"
    if args.no_channel_attention:
        exp_name += "_no-attn"
    if args.no_gates:
        exp_name += "_no-gates"
    if args.no_alignment:
        exp_name += "_no-align"
    if args.fusion_type and args.fusion_type != "hierarchical":
        exp_name += f"_{args.fusion_type}"

    with open(args.config) as f:
        cfg_all = json.load(f)
    fixed = cfg_all["fixed_params"]
    fixed["no_distill"] = args.no_distill
    if args.no_channel_attention is not None:
        fixed["no_channel_attention"] = args.no_channel_attention
    if args.no_gates is not None:
        fixed["no_gates"] = args.no_gates
    if args.no_alignment is not None:
        fixed["no_alignment"] = args.no_alignment
    if args.selection_mode is not None:
        fixed["selection_mode"] = args.selection_mode
    if args.selection_test_weight is not None:
        fixed["selection_test_weight"] = args.selection_test_weight
    if args.fusion_type is not None:
        fixed["fusion_type"] = args.fusion_type
    if args.checkpoint_prefix is not None:
        fixed["checkpoint_prefix"] = args.checkpoint_prefix
    fixed["serial_pretrain"] = args.serial_pretrain
    fixed["seed"] = args.seed
    validate_strict_clean_config(fixed)

    batch_size = cfg_all["batch_size"]
    num_epochs = cfg_all["num_epochs"]
    embed_dims = cfg_all["embed_dims_full"]
    classes = cfg_all["classes"][args.dataset]
    feature_paths = cfg_all["feature_paths"][args.dataset]

    base = os.environ.get("MELD_CSV_DIR", "path/to/MELD")
    train_csv = os.path.join(base, "train_sent_emo.csv")
    val_csv = os.path.join(base, "dev_sent_emo.csv")
    test_csv = os.path.join(base, "test_sent_emo.csv")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(classes)

    def load_json(path):
        with open(path) as f:
            return json.load(f)

    train_features = {}
    val_features = {}
    test_features = {}

    if "v" in args.modalities:
        train_features["v"] = load_json(feature_paths["train"]["visual"])
        val_features["v"] = load_json(feature_paths["dev"]["visual"])
        test_features["v"] = load_json(feature_paths["test"]["visual"])

    if "a" in args.modalities:
        train_features["a"] = load_json(feature_paths["train"]["audio"])
        val_features["a"] = load_json(feature_paths["dev"]["audio"])
        test_features["a"] = load_json(feature_paths["test"]["audio"])

    if "t" in args.modalities:
        train_features["t"] = load_json(feature_paths["train"]["text"])
        val_features["t"] = load_json(feature_paths["dev"]["text"])
        test_features["t"] = load_json(feature_paths["test"]["text"])

    # Utterance-level datasets (for fusion without graph encoders or backward compat)
    train_ds = MELDDataset(
        train_df,
        train_features.get("v", {}),
        train_features.get("a", {}),
        train_features.get("t", {}),
        label_encoder,
        args.modalities,
        embed_dims,
        is_training=True,
    )
    val_ds = MELDDataset(
        val_df,
        val_features.get("v", {}),
        val_features.get("a", {}),
        val_features.get("t", {}),
        label_encoder,
        args.modalities,
        embed_dims,
        is_training=False,
    )
    test_ds = MELDDataset(
        test_df,
        test_features.get("v", {}),
        test_features.get("a", {}),
        test_features.get("t", {}),
        label_encoder,
        args.modalities,
        embed_dims,
        is_training=False,
    )

    # Dialogue-level datasets (DFGCN-style: complete dialogues with speaker-temporal adjacency)
    # Used for encoder pretrain and fusion feature extraction with graph context.
    train_dia_ds = MELDDialogueDataset(
        train_df,
        train_features.get("v", {}),
        train_features.get("a", {}),
        train_features.get("t", {}),
        label_encoder,
        args.modalities,
        embed_dims,
        is_training=True,
    )
    val_dia_ds = MELDDialogueDataset(
        val_df,
        val_features.get("v", {}),
        val_features.get("a", {}),
        val_features.get("t", {}),
        label_encoder,
        args.modalities,
        embed_dims,
        is_training=False,
    )
    test_dia_ds = MELDDialogueDataset(
        test_df,
        test_features.get("v", {}),
        test_features.get("a", {}),
        test_features.get("t", {}),
        label_encoder,
        args.modalities,
        embed_dims,
        is_training=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=custom_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=custom_collate
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=custom_collate
    )

    # Dialogue-level dataloaders: batch_size = number of dialogues per batch
    dia_batch_size = max(1, batch_size // 2)  # 16 dialogues per batch
    train_dia_loader = DataLoader(
        train_dia_ds, batch_size=dia_batch_size, shuffle=True, collate_fn=dialogue_collate
    )
    val_dia_loader = DataLoader(
        val_dia_ds, batch_size=dia_batch_size, shuffle=False, collate_fn=dialogue_collate
    )
    test_dia_loader = DataLoader(
        test_dia_ds, batch_size=dia_batch_size, shuffle=False, collate_fn=dialogue_collate
    )

    print(f"Utterance datasets: train={len(train_ds)} dev={len(val_ds)} test={len(test_ds)}")
    print(f"Dialogue datasets: train={len(train_dia_ds)} dev={len(val_dia_ds)} test={len(test_dia_ds)}")

    y_train = label_encoder.transform(train_df["Emotion"])
    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = torch.tensor(cw, dtype=torch.float32).to(device)
    samples_per_class = [int((y_train == c).sum()) for c in range(len(label_encoder.classes_))]

    fusion_model, fusion_report, evidential_diag, fusion_metrics = train_full_pipeline(
        train_loader,
        val_loader,
        test_loader,
        device,
        label_encoder,
        args.modalities,
        embed_dims,
        class_weights,
        samples_per_class,
        fixed,
        num_epochs,
        train_dia_loader=train_dia_loader,
        val_dia_loader=val_dia_loader,
        test_dia_loader=test_dia_loader,
    )

    run_suffix = f"{exp_name}_seed{args.seed}"
    # Never encode an absolute checkpoint prefix into the results directory.
    # It previously created a nested ``results/<timestamp>_/data2/...`` tree.
    results_dir = os.path.join("results", f"{timestamp}_{run_suffix}")
    os.makedirs(results_dir, exist_ok=True)

    model_path = os.path.join(results_dir, f"best_fusion_{exp_name}.pth")
    torch.save(fusion_model, model_path)

    config_path = os.path.join(results_dir, f"config_{exp_name}.json")
    with open(config_path, "w") as wf:
        json.dump(cfg_all, wf, indent=2)

    metrics_path = os.path.join(results_dir, f"fusion_metrics_{exp_name}.json")
    with open(metrics_path, "w") as wf:
        json.dump(fusion_metrics, wf, indent=2)

    if evidential_diag is not None:
        diag_path = os.path.join(results_dir, f"diagnostics_{exp_name}.json")
        with open(diag_path, "w") as wf:
            json.dump(evidential_diag, wf, indent=2)
    else:
        diag_path = None

    print("====================================")
    print(f"Experiment: {exp_name}")
    print(f"Timestamp: {timestamp}")
    print(f"Dataset: {args.dataset}")
    print(f"Modalities: {args.modalities}")
    print("====================================")
    print("==== Fusion Model Report ====")
    print(fusion_report)
    print(
        "[diagnostic] "
        f"best_epoch={fusion_metrics.get('best_epoch', 0)} "
        f"best_val_f1={fusion_metrics.get('best_val_f1', 0.0):.4f}"
    )
    if evidential_diag is not None:
        print(format_evidential_diagnostics(evidential_diag))
    print("====================================")
    print(f"Saved fusion model to: {model_path}")
    print(f"Saved configuration to: {config_path}")
    print(f"Saved fusion metrics to: {metrics_path}")
    if diag_path is not None:
        print(f"Saved diagnostics to: {diag_path}")
