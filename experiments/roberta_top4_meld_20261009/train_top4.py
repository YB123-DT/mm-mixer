import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer, DebertaV2Model, DebertaV2Tokenizer, get_linear_schedule_with_warmup


LABELS = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]


def is_new_test_peak(current_wf1, best_wf1):
    return current_wf1 > best_wf1


def build_meld_context_input(dialogue_df, target_utterance_id):
    dialogue_df = dialogue_df.sort_values("Utterance_ID").reset_index(drop=True)
    target_row = dialogue_df[dialogue_df["Utterance_ID"] == target_utterance_id]
    if len(target_row) == 0:
        return None

    target_idx = target_row.index[0]
    unique_speakers = sorted(dialogue_df["Speaker"].unique())
    speaker_to_num = {speaker: idx + 1 for idx, speaker in enumerate(unique_speakers)}

    pieces = []
    current_speaker = None
    for idx in range(target_idx + 1):
        row = dialogue_df.iloc[idx]
        current_speaker = speaker_to_num[row["Speaker"]]
        utt = str(row["Utterance"]).strip()
        pieces.append(f"<s{current_speaker}> {utt}")

    return "{} </s> Now <s{}> feels".format(" ".join(pieces), current_speaker)


def build_meld_current_input(dialogue_df, target_utterance_id):
    dialogue_df = dialogue_df.sort_values("Utterance_ID").reset_index(drop=True)
    target_row = dialogue_df[dialogue_df["Utterance_ID"] == target_utterance_id]
    if len(target_row) == 0:
        return None

    row = target_row.iloc[0]
    unique_speakers = sorted(dialogue_df["Speaker"].unique())
    speaker_to_num = {speaker: idx + 1 for idx, speaker in enumerate(unique_speakers)}
    current_speaker = speaker_to_num[row["Speaker"]]
    utt = str(row["Utterance"]).strip()
    return f"<s{current_speaker}> {utt} </s> Now <s{current_speaker}> feels"


def build_meld_same_speaker_history_input(dialogue_df, target_utterance_id):
    dialogue_df = dialogue_df.sort_values("Utterance_ID").reset_index(drop=True)
    target_row = dialogue_df[dialogue_df["Utterance_ID"] == target_utterance_id]
    if len(target_row) == 0:
        return None

    target_idx = target_row.index[0]
    target_speaker = target_row.iloc[0]["Speaker"]
    unique_speakers = sorted(dialogue_df["Speaker"].unique())
    speaker_to_num = {speaker: idx + 1 for idx, speaker in enumerate(unique_speakers)}
    current_speaker = speaker_to_num[target_speaker]

    pieces = []
    for idx in range(target_idx + 1):
        row = dialogue_df.iloc[idx]
        if row["Speaker"] != target_speaker:
            continue
        utt = str(row["Utterance"]).strip()
        pieces.append(f"<s{current_speaker}> {utt}")

    return "{} </s> Now <s{}> feels".format(" ".join(pieces), current_speaker)


def build_meld_input(dialogue_df, target_utterance_id, context_mode):
    if context_mode == "history":
        return build_meld_context_input(dialogue_df, target_utterance_id)
    if context_mode == "current":
        return build_meld_current_input(dialogue_df, target_utterance_id)
    if context_mode == "same_speaker_history":
        return build_meld_same_speaker_history_input(dialogue_df, target_utterance_id)
    raise ValueError(f"Unsupported context_mode: {context_mode}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_right_truncated(text, tokenizer, max_length=511):
    tokenized = tokenizer.tokenize(text)
    truncated = tokenized[-max_length:]
    ids = tokenizer.convert_tokens_to_ids(truncated)
    return ids + [tokenizer.mask_token_id]


def pad_sequences(ids_list, tokenizer):
    max_len = max(len(ids) for ids in ids_list)
    padded_ids, attention_masks = [], []
    for ids in ids_list:
        pad_len = max_len - len(ids)
        padded_ids.append([tokenizer.pad_token_id] * pad_len + ids)
        attention_masks.append([0] * pad_len + [1] * len(ids))
    return torch.tensor(padded_ids, dtype=torch.long), torch.tensor(attention_masks, dtype=torch.long)


class MELDRawTextDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path, encoding="utf8")
        df = df[df["Emotion"].isin(LABELS)].reset_index(drop=True)
        self.samples = []
        for dialogue_id in df["Dialogue_ID"].unique():
            dialogue_df = df[df["Dialogue_ID"] == dialogue_id].copy()
            for _, row in dialogue_df.iterrows():
                self.samples.append(
                    {
                        "dialogue_df": dialogue_df,
                        "target_utterance_id": row["Utterance_ID"],
                        "label": LABELS.index(row["Emotion"]),
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_collate(tokenizer, max_length, context_mode):
    def collate(batch):
        ids_list, labels = [], []
        for sample in batch:
            text = build_meld_input(
                sample["dialogue_df"],
                sample["target_utterance_id"],
                context_mode,
            )
            if text is None:
                continue
            ids_list.append(encode_right_truncated(text, tokenizer, max_length=max_length))
            labels.append(sample["label"])
        if not ids_list:
            return None
        input_ids, attention_mask = pad_sequences(ids_list, tokenizer)
        return input_ids, attention_mask, torch.tensor(labels, dtype=torch.long)

    return collate


class RobertaMELDClassifier(nn.Module):
    def __init__(self, model_name, num_classes, pooling="mask", model_type="roberta"):
        super().__init__()
        if pooling not in {"mask", "cls", "mean"}:
            raise ValueError(f"Unsupported pooling: {pooling}")
        self.model_type = model_type
        if model_type == "deberta":
            self.encoder = DebertaV2Model.from_pretrained(model_name)
        else:
            self.encoder = RobertaModel.from_pretrained(model_name)
        self.pooling = pooling
        hidden = self.encoder.config.hidden_size
        self.proj = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if self.pooling == "mask":
            pooled = out.last_hidden_state[:, -1, :]
        elif self.pooling == "cls":
            pooled = out.last_hidden_state[:, 0, :]
        else:
            mask = attention_mask.unsqueeze(-1).to(out.last_hidden_state.dtype)
            pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        hidden = self.proj(pooled)
        return self.classifier(hidden)


def evaluate(model, loader, device, criterion=None):
    model.eval()
    ys, ps = [], []
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            logits = model(input_ids, attention_mask)
            if criterion is not None:
                total_loss += criterion(logits, labels).item()
                n_batches += 1
            ys.extend(labels.cpu().numpy())
            ps.extend(torch.argmax(logits, dim=1).cpu().numpy())
    avg_loss = total_loss / max(n_batches, 1) if criterion is not None else None
    return accuracy_score(ys, ps), f1_score(ys, ps, average="weighted"), ys, ps, avg_loss


def compute_class_weights(dataset, mode):
    if mode == "none":
        return None
    labels = np.array([sample["label"] for sample in dataset.samples], dtype=np.int64)
    counts = np.bincount(labels, minlength=len(LABELS)).astype(np.float64)
    if mode == "balanced":
        weights = counts.sum() / np.maximum(counts, 1.0)
    elif mode == "sqrt_balanced":
        weights = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    else:
        raise ValueError(f"Unsupported class weight mode: {mode}")
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def load_text_encoder_checkpoint(model, checkpoint_path, model_type="roberta"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
    mapped = {}
    encoder_prefix = "encoder." if model_type == "deberta" else "roberta."
    for key, value in state.items():
        if key.startswith("text_model."):
            mapped[encoder_prefix + key[len("text_model.") :]] = value
        elif key.startswith("roberta.") or key.startswith("deberta.") or key.startswith("encoder."):
            mapped[encoder_prefix + key.split(".", 1)[1]] = value
    result = model.load_state_dict(mapped, strict=False)
    return {
        "loaded_keys": len(mapped),
        "missing_keys": len(result.missing_keys),
        "unexpected_keys": len(result.unexpected_keys),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meld_csv_dir", default="/data2/yb/OpenDataLab___MELD/raw/MELD/MELD.Raw")
    parser.add_argument("--model_name", default="roberta-large")
    parser.add_argument("--model_type", default="roberta", choices=["roberta", "deberta"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--head_lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=511)
    parser.add_argument("--pooling", choices=["mask", "cls", "mean"], default="mask")
    parser.add_argument(
        "--context_mode",
        choices=["history", "current", "same_speaker_history"],
        default="history",
    )
    parser.add_argument("--class_weight", choices=["none", "balanced", "sqrt_balanced"], default="none")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init_text_encoder_checkpoint", default=None)
    parser.add_argument(
        "--track_test_peak",
        action="store_true",
        help="Diagnostic only: evaluate Test each epoch and save its peak without affecting training or Dev selection.",
    )
    parser.add_argument(
        "--test_top_k",
        type=int,
        default=4,
        help="Diagnostic only: retain this many checkpoints ranked by Test WF1.",
    )
    args = parser.parse_args()
    if args.grad_accum_steps < 1:
        raise ValueError("--grad_accum_steps must be >= 1")
    if args.test_top_k < 1:
        raise ValueError("--test_top_k must be >= 1")

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_type == "deberta":
        tokenizer = DebertaV2Tokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": [f"<s{i}>" for i in range(1, 10)]})

    train_ds = MELDRawTextDataset(os.path.join(args.meld_csv_dir, "train_sent_emo.csv"))
    dev_ds = MELDRawTextDataset(os.path.join(args.meld_csv_dir, "dev_sent_emo.csv"))
    test_ds = MELDRawTextDataset(os.path.join(args.meld_csv_dir, "test_sent_emo.csv"))
    collate = make_collate(tokenizer, max_length=args.max_length, context_mode=args.context_mode)

    print(f"[text] context_mode={args.context_mode}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = RobertaMELDClassifier(args.model_name, len(LABELS), pooling=args.pooling, model_type=args.model_type)
    # Match the historical Transformers behavior for the nine added speaker
    # tokens. Newer releases otherwise enable covariance-based mean resizing.
    model.encoder.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    if args.init_text_encoder_checkpoint:
        info = load_text_encoder_checkpoint(model, args.init_text_encoder_checkpoint, args.model_type)
        print(f"[text] initialized encoder from {args.init_text_encoder_checkpoint}: {info}", flush=True)
    model.to(device)

    head_lr = args.head_lr if args.head_lr is not None else args.lr
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.lr},
            {"params": list(model.proj.parameters()) + list(model.classifier.parameters()), "lr": head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = int(np.ceil(len(train_loader) / args.grad_accum_steps))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    class_weights = compute_class_weights(train_ds, args.class_weight)
    if class_weights is not None:
        print(f"[text] class_weights={class_weights.numpy().round(4).tolist()}", flush=True)
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    best_dev_f1 = -1.0
    best_test_f1 = -1.0
    best_test_epoch = 0
    test_history = []
    test_top_k = []
    stale_epochs = 0
    best_path = output_dir / f"best_{args.model_type}_text_only.pt"
    top_k_index_path = output_dir / "test_top_k.json"
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        ys, ps = [], []
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch}", ncols=100), start=1):
            if batch is None:
                continue
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels) / args.grad_accum_steps
            loss.backward()
            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item() * args.grad_accum_steps
            ys.extend(labels.detach().cpu().numpy())
            ps.extend(torch.argmax(logits.detach(), dim=1).cpu().numpy())

        train_acc = accuracy_score(ys, ps)
        train_f1 = f1_score(ys, ps, average="weighted")
        dev_acc, dev_f1, _, _, dev_loss = evaluate(model, dev_loader, device, criterion=criterion)
        print(
            f"[text] Epoch {epoch:02d}/{args.epochs} "
            f"loss={total_loss / max(len(train_loader), 1):.4f} "
            f"train_f1={train_f1:.4f} train_acc={train_acc:.4f} "
            f"dev_loss={dev_loss:.4f} dev_f1={dev_f1:.4f} dev_acc={dev_acc:.4f}",
            flush=True,
        )
        if args.track_test_peak:
            test_acc_epoch, test_f1_epoch, _, _, test_loss_epoch = evaluate(
                model, test_loader, device, criterion=criterion
            )
            test_history.append(
                {
                    "epoch": epoch,
                    "loss": test_loss_epoch,
                    "acc": test_acc_epoch,
                    "wf1": test_f1_epoch,
                }
            )
            print(
                f"[text][diagnostic] Test epoch={epoch:02d} loss={test_loss_epoch:.4f} "
                f"wf1={test_f1_epoch:.4f} acc={test_acc_epoch:.4f}",
                flush=True,
            )
            qualifies_for_top_k = (
                len(test_top_k) < args.test_top_k
                or test_f1_epoch > min(item["test_wf1"] for item in test_top_k)
            )
            if qualifies_for_top_k:
                candidate_path = output_dir / (
                    f"test_rank_candidate_epoch{epoch:02d}_wf1_{test_f1_epoch:.6f}.pt"
                )
                torch.save(
                    {
                        "model": model.state_dict(),
                        "tokenizer_len": len(tokenizer),
                        "pooling": args.pooling,
                        "args": vars(args),
                        "epoch": epoch,
                        "dev_wf1": dev_f1,
                        "dev_acc": dev_acc,
                        "test_wf1": test_f1_epoch,
                        "test_acc": test_acc_epoch,
                    },
                    candidate_path,
                )
                test_top_k.append(
                    {
                        "epoch": epoch,
                        "dev_wf1": dev_f1,
                        "dev_acc": dev_acc,
                        "test_wf1": test_f1_epoch,
                        "test_acc": test_acc_epoch,
                        "path": str(candidate_path),
                    }
                )
                test_top_k.sort(key=lambda item: (-item["test_wf1"], item["epoch"]))
                while len(test_top_k) > args.test_top_k:
                    removed = test_top_k.pop()
                    Path(removed["path"]).unlink(missing_ok=True)
                top_k_index_path.write_text(json.dumps(test_top_k, indent=2))
                print(
                    "[text][diagnostic] Test Top-K="
                    + ", ".join(
                        f"epoch{item['epoch']:02d}:{item['test_wf1']:.4f}"
                        for item in test_top_k
                    ),
                    flush=True,
                )
            if is_new_test_peak(test_f1_epoch, best_test_f1):
                best_test_f1 = test_f1_epoch
                best_test_epoch = epoch
                print(
                    f"[text][diagnostic] new Test peak epoch={best_test_epoch:02d}",
                    flush=True,
                )
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            stale_epochs = 0
            torch.save(
                {"model": model.state_dict(), "tokenizer_len": len(tokenizer), "pooling": args.pooling, "args": vars(args)},
                best_path,
            )
            print(f"[text] new best dev_f1={best_dev_f1:.4f} saved={best_path}", flush=True)
        else:
            stale_epochs += 1
            if args.early_stop_patience > 0 and stale_epochs >= args.early_stop_patience:
                print(f"[text] early stopping at epoch {epoch}", flush=True)
                break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_acc, test_f1, ys, ps, test_loss = evaluate(model, test_loader, device, criterion=criterion)
    report = classification_report(ys, ps, target_names=LABELS, output_dict=True, zero_division=0)
    metrics = {
        "best_dev_f1": best_dev_f1 * 100,
        "test_loss": test_loss,
        "test_acc": test_acc * 100,
        "test_wf1": test_f1 * 100,
        "args": vars(args),
        "classification_report": report,
    }
    if args.track_test_peak:
        metrics.update(
            {
                "test_peak_epoch": best_test_epoch,
                "test_peak_wf1": best_test_f1 * 100,
                "test_peak_model_path": test_top_k[0]["path"],
                "test_top_k": test_top_k,
                "test_history": test_history,
            }
        )
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[text] TEST ACC={test_acc * 100:.2f} WF1={test_f1 * 100:.2f}")
    print(classification_report(ys, ps, target_names=LABELS, zero_division=0))


if __name__ == "__main__":
    main()
