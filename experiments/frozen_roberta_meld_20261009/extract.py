"""Extract task-agnostic frozen RoBERTa-large features for MELD.

This is a controlled diagnostic: it keeps the existing causal history string,
right truncation, and last-non-padding (normally EOS) pooling, while loading no
MELD-fine-tuned checkpoint.  Speaker markers are tokenized by the original
RoBERTa tokenizer so that no random embedding rows are introduced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import RobertaModel, RobertaTokenizer


LABELS = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def history_text(dialogue: pd.DataFrame, target_utterance_id: int) -> str:
    dialogue = dialogue.sort_values("Utterance_ID").reset_index(drop=True)
    target = dialogue.index[dialogue["Utterance_ID"] == target_utterance_id]
    if len(target) != 1:
        raise ValueError(f"expected one target utterance {target_utterance_id}")
    speakers = sorted(dialogue["Speaker"].unique())
    speaker_number = {speaker: index + 1 for index, speaker in enumerate(speakers)}
    pieces = []
    current = None
    for index in range(int(target[0]) + 1):
        row = dialogue.iloc[index]
        current = speaker_number[row["Speaker"]]
        pieces.append(f"<s{current}> {str(row['Utterance']).strip()}")
    return f"{' '.join(pieces)} </s> Now <s{current}> feels"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=511)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)

    device = torch.device("cuda")
    tokenizer = RobertaTokenizer.from_pretrained(args.model, local_files_only=True)
    model = RobertaModel.from_pretrained(args.model, local_files_only=True).to(device).eval()
    original_vocabulary = len(tokenizer)
    original_embeddings = model.get_input_embeddings().num_embeddings
    if original_vocabulary != original_embeddings:
        raise RuntimeError((original_vocabulary, original_embeddings))

    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "kind": "task_agnostic_frozen_roberta_large",
        "model": args.model,
        "seed": args.seed,
        "pooling": "last_non_padding_normally_eos",
        "context": "history_through_current_utterance",
        "truncation": "right",
        "max_length": args.max_length,
        "speaker_markers": "original_tokenizer_no_added_tokens",
        "tokenizer_length": original_vocabulary,
        "embedding_rows": original_embeddings,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "splits": {},
    }

    for split in ("train", "dev", "test"):
        csv_path = args.csv_dir / f"{split}_sent_emo.csv"
        frame = pd.read_csv(csv_path)
        dialogues = {
            int(dialogue_id): part.copy()
            for dialogue_id, part in frame.groupby("Dialogue_ID", sort=False)
        }
        records = [
            (f"dia{int(row.Dialogue_ID)}_utt{int(row.Utterance_ID)}",
             history_text(dialogues[int(row.Dialogue_ID)], int(row.Utterance_ID)))
            for row in frame.itertuples()
        ]
        features: dict[str, list[float]] = {}
        length_over_limit = 0
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            raw_lengths = tokenizer(
                [text for _, text in batch], add_special_tokens=True,
                truncation=False, padding=False,
            )["input_ids"]
            length_over_limit += sum(len(ids) > args.max_length for ids in raw_lengths)
            encoded = tokenizer(
                [text for _, text in batch], padding=True, truncation=True,
                max_length=args.max_length, return_tensors="pt",
            )
            encoded = {name: value.to(device) for name, value in encoded.items()}
            with torch.inference_mode():
                hidden = model(**encoded).last_hidden_state
                indices = encoded["attention_mask"].sum(dim=1) - 1
                pooled = hidden[torch.arange(len(batch), device=device), indices]
            pooled = pooled.float().cpu().numpy()
            for (key, _), vector in zip(batch, pooled):
                features[key] = vector.tolist()
            print(f"{split}: {min(start + len(batch), len(records))}/{len(records)}", flush=True)

        split_dir = args.output / f"{split}_features"
        split_dir.mkdir()
        output_path = split_dir / "text_features.json"
        output_path.write_text(json.dumps(features))
        matrix = np.asarray(list(features.values()), dtype=np.float32)
        manifest["splits"][split] = {
            "rows": len(features),
            "dimension": int(matrix.shape[1]),
            "finite": bool(np.isfinite(matrix).all()),
            "over_max_length": length_over_limit,
            "csv": str(csv_path),
            "csv_sha256": sha256(csv_path),
            "output": str(output_path),
            "output_sha256": sha256(output_path),
        }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
