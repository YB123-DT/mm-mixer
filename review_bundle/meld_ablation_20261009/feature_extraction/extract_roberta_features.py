"""Extract RoBERTa-large (1024-dim) features for MELD utterances.

Mirrors the context format from train_roberta_text_only_meld.py but outputs
features instead of classification logits. Saves in the same JSON format as
the existing text_features.json files.
"""
import argparse
import json
import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaModel, RobertaTokenizer
from tqdm import tqdm

LABELS = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]


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
    context = " ".join(pieces)
    return f"{context} </s> Now <s{current_speaker}> feels"


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
    return f"{' '.join(pieces)} </s> Now <s{current_speaker}> feels"


def build_meld_input(dialogue_df, target_utterance_id, context_mode):
    if context_mode == "history":
        return build_meld_context_input(dialogue_df, target_utterance_id)
    if context_mode == "current":
        return build_meld_current_input(dialogue_df, target_utterance_id)
    if context_mode == "same_speaker_history":
        return build_meld_same_speaker_history_input(dialogue_df, target_utterance_id)
    raise ValueError(f"Unsupported context_mode: {context_mode}")


class MELDRawTextDataset(Dataset):
    def __init__(self, csv_path, context_mode):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.context_mode = context_mode
        self.dialogues = {}
        self.samples = []
        for _, row in self.df.iterrows():
            dia_id = row["Dialogue_ID"]
            utt_id = row["Utterance_ID"]
            if dia_id not in self.dialogues:
                self.dialogues[dia_id] = []
            self.dialogues[dia_id].append(row)
            self.samples.append((dia_id, utt_id, LABELS.index(row["Emotion"])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dia_id, utt_id, label = self.samples[idx]
        dialogue_df = self.df[self.df["Dialogue_ID"] == dia_id]
        text = build_meld_input(dialogue_df, utt_id, self.context_mode)
        return dia_id, utt_id, text, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meld_csv_dir", default="/data2/yb/OpenDataLab___MELD/raw/MELD/MELD.Raw")
    parser.add_argument("--model_name", default="roberta-large")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=511)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default=None, help="Fine-tuned checkpoint to load")
    parser.add_argument(
        "--context_mode",
        choices=["history", "current", "same_speaker_history"],
        default=None,
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[extract] Loading {args.model_name} on {device}...")
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": [f"<s{i}>" for i in range(1, 10)]})

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"[extract] Loading fine-tuned checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        checkpoint_args = ckpt.get("args", {})
        if args.context_mode is None:
            args.context_mode = checkpoint_args.get("context_mode", "history")
        model = RobertaModel.from_pretrained(args.model_name)
        tokenizer_len = ckpt.get("tokenizer_len")
        if tokenizer_len is None and "tokenizer" in ckpt:
            try:
                tokenizer_len = len(ckpt["tokenizer"])
            except Exception:
                tokenizer_len = len(tokenizer)
        if tokenizer_len is None:
            tokenizer_len = len(tokenizer)
        model.resize_token_embeddings(tokenizer_len)

        roberta_state = {}
        raw_state = ckpt.get("model")
        if raw_state is None:
            raw_state = ckpt.get("model_state_dict", ckpt)

        for k, v in raw_state.items():
            if k.startswith("roberta."):
                roberta_state[k[len("roberta."):]] = v
            elif k.startswith("encoder."):
                roberta_state[k[len("encoder."):]] = v
            elif k.startswith("text_model."):
                roberta_state[k[len("text_model."):]] = v
            elif not "." in k:
                roberta_state[k] = v
        if not roberta_state:
            raise ValueError(
                f"No RoBERTa encoder weights matched in checkpoint: {args.checkpoint}"
            )
        model.load_state_dict(roberta_state, strict=False)
        print(f"[extract] Loaded {len(roberta_state)} RoBERTa weights from checkpoint")
    else:
        print(f"[extract] Using pretrained {args.model_name} (no fine-tuning checkpoint)")
        model = RobertaModel.from_pretrained(args.model_name)
        model.resize_token_embeddings(len(tokenizer))
        if args.context_mode is None:
            args.context_mode = "history"

    model.to(device)
    model.eval()
    hidden_dim = model.config.hidden_size
    print(f"[extract] Hidden dim: {hidden_dim}")
    print(f"[extract] context_mode: {args.context_mode}")

    for split in ["train", "dev", "test"]:
        csv_path = os.path.join(args.meld_csv_dir, f"{split}_sent_emo.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(args.meld_csv_dir, f"{split}_sent_emo.csv")
        print(f"[extract] Processing {split} split: {csv_path}")
        ds = MELDRawTextDataset(csv_path, args.context_mode)
        print(f"[extract]   {len(ds)} utterances")

        features = {}
        # Process in batches
        for start in tqdm(range(0, len(ds), args.batch_size), desc=f"  {split}"):
            end = min(start + args.batch_size, len(ds))
            batch_texts = []
            batch_keys = []
            for j in range(start, end):
                dia_id, utt_id, text, _ = ds[j]
                batch_keys.append((dia_id, utt_id))
                batch_texts.append(text if text else "")

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            with torch.no_grad():
                out_hidden = model(input_ids=input_ids, attention_mask=attention_mask)
                # mask pooling: last non-padding token
                last_idx = attention_mask.sum(dim=1) - 1  # (B,)
                pooled = out_hidden.last_hidden_state[torch.arange(len(last_idx)), last_idx, :]  # (B, D)
                pooled = pooled.cpu().numpy()

            for j, (dia_id, utt_id) in enumerate(batch_keys):
                key = f"dia{dia_id}_utt{utt_id}"
                features[key] = pooled[j].tolist()

        split_dir = out / f"{split}_features"
        split_dir.mkdir(parents=True, exist_ok=True)
        out_file = split_dir / "text_features.json"
        with open(out_file, "w") as f:
            json.dump(features, f)
        print(f"[extract]   Saved {len(features)} features to {out_file}")

    print("[extract] Done.")


if __name__ == "__main__":
    main()
