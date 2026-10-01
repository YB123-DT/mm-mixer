"""Build evidence-grounded MELD examples and an implementation-grounded AMM diagram."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import wave

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams.update({
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "savefig.facecolor": "white",
})

BLUE = "#0F4D92"
TEAL = "#42949E"
VIOLET = "#9A4D8E"
INK = "#272727"
PDF_DIRECTORY = Path(__file__).resolve().parents[2]
CASES = (
    # Exact current utterance and speaker; frames checked for target visibility.
    ("train", 645, 13, "joy", 23),
    ("test", 163, 2, "sadness", 26),
    ("train", 700, 0, "neutral", 7),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cases(raw_root: Path, audio_root: Path, out: Path):
    images, waveforms, records = [], [], []
    for split, dialogue, utterance, emotion, frame_index in CASES:
        csv_path = raw_root / f"{split}_sent_emo.csv"
        metadata = pd.read_csv(csv_path)
        selected = metadata[
            (metadata.Dialogue_ID == dialogue) & (metadata.Utterance_ID == utterance)
        ]
        if len(selected) != 1:
            raise ValueError(f"Expected one row for {split}/{dialogue}/{utterance}")
        row = selected.iloc[0]
        if (row.Utterance, row.Speaker, row.Emotion) != ("Hey!", "Ross", emotion):
            raise ValueError("Source annotations disagree with the selected example")
        key = f"dia{dialogue}_utt{utterance}"
        video_dir = "train_splits" if split == "train" else "output_repeated_splits_test"
        video_path = raw_root / video_dir / f"{key}.mp4"
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise FileNotFoundError(video_path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not 0 <= frame_index < frame_count or fps <= 0:
            capture.release()
            raise ValueError(f"Invalid video/frame metadata: {video_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, bgr = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError(f"Could not decode frame {frame_index}: {video_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image_path = out / "source_frames" / f"{split}_{key}_frame{frame_index}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(image_path, rgb)

        audio_path = audio_root / f"{split}_audio" / f"{key}.wav"
        with wave.open(str(audio_path), "rb") as handle:
            if (handle.getnchannels(), handle.getsampwidth()) != (1, 2):
                raise ValueError("Expected the existing mono 16-bit PCM clip audio")
            rate = handle.getframerate()
            samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
        if rate != 16000 or len(samples) == 0:
            raise ValueError("Expected nonempty 16 kHz clip audio")
        audio = samples.astype(np.float64) / 32768.0
        preceding = metadata[
            (metadata.Dialogue_ID == dialogue)
            & metadata.Utterance_ID.between(max(utterance - 2, 0), utterance - 1)
        ]
        records.append({
            "split": split,
            "sample_id": key,
            "speaker": str(row.Speaker),
            "current_utterance": str(row.Utterance),
            "dataset_emotion": str(row.Emotion),
            "csv_row_index": int(selected.index[0]),
            "csv_source": str(csv_path),
            "csv_sha256": sha256(csv_path),
            "video_source": str(video_path),
            "video_sha256": sha256(video_path),
            "video_frame_count": frame_count,
            "video_fps": fps,
            "frame_index": frame_index,
            "frame_time_seconds": frame_index / fps,
            "frame_file": str(image_path.relative_to(out)),
            "frame_sha256": sha256(image_path),
            "audio_source": str(audio_path),
            "audio_sha256": sha256(audio_path),
            "audio_sample_rate": rate,
            "audio_sample_count": len(samples),
            "audio_duration_seconds": len(samples) / rate,
            "preceding_two_utterances": json.dumps(
                preceding[["Speaker", "Utterance", "Emotion", "Utterance_ID"]]
                .to_dict("records"), ensure_ascii=False
            ),
            "selection": "same speaker and exact current words; distinct labels; target visible",
        })
        images.append(rgb)
        waveforms.append((rate, audio))
    return images, waveforms, records


def export(fig, out: Path, name: str):
    fig.canvas.draw()
    bounds = fig.bbox
    for item in fig.findobj(match=matplotlib.text.Text):
        if not item.get_visible() or not item.get_text():
            continue
        box = item.get_window_extent(fig.canvas.get_renderer())
        if box.x0 < bounds.x0 - 1 or box.x1 > bounds.x1 + 1:
            raise RuntimeError(f"Text clipped horizontally: {item.get_text()}")
        if box.y0 < bounds.y0 - 1 or box.y1 > bounds.y1 + 1:
            raise RuntimeError(f"Text clipped vertically: {item.get_text()}")
    for suffix in ("svg", "pdf", "png"):
        destination = PDF_DIRECTORY if suffix == "pdf" else out
        fig.savefig(destination / f"{name}.{suffix}", dpi=450)
    plt.close(fig)


def scene_figure(images, waveforms, records, out: Path):
    fig = plt.figure(figsize=(180 / 25.4, 100 / 25.4))
    fig.text(.5, .945, "Same words, different annotated emotions", ha="center",
             fontsize=10, fontweight="bold", color=INK)
    fig.text(.5, .875, 'Same speaker: Ross    |    Same current utterance: "Hey!"',
             ha="center", fontsize=8, color=INK)
    duration = max(len(audio) / rate for rate, audio in waveforms)
    for i, (rgb, (rate, audio), row) in enumerate(zip(images, waveforms, records)):
        left = .048 + i * .324
        center = left + .14
        fig.text(center, .79, f"{chr(97 + i)}   {row['dataset_emotion'].capitalize()}",
                 ha="center", fontsize=9, fontweight="bold", color=INK)
        ax = fig.add_axes([left, .435, .28, .30])
        ax.imshow(rgb)
        ax.set_axis_off()
        fig.text(center, .399, f"{row['split']} / {row['sample_id']}",
                 ha="center", fontsize=6.5, color=INK)
        ax = fig.add_axes([left, .177, .28, .145])
        # Min/max envelope of fixed 64-sample bins; common PCM scale and time axis.
        starts = np.arange(0, len(audio), 64)
        lower = np.minimum.reduceat(audio, starts)
        upper = np.maximum.reduceat(audio, starts)
        time = starts / rate
        ax.fill_between(time, lower, upper, color=BLUE, linewidth=.25)
        ax.set(xlim=(0, duration), ylim=(-1, 1), xticks=[0, 1, 2], yticks=[-1, 0, 1])
        ax.set_xlabel("Time (s)", fontsize=7, labelpad=2)
        if i == 0:
            ax.set_ylabel("PCM amplitude", fontsize=7, labelpad=1)
        else:
            ax.set_yticklabels([])
        ax.set_title("Clip audio envelope", fontsize=7, pad=3)
        ax.tick_params(length=2, width=.5, pad=2)
    fig.text(.5, .059,
             "MELD / Friends. Illustrative source clips; labels are dataset annotations.",
             ha="center", fontsize=6.5, color=INK)
    fig.text(.5, .026,
             "Identical current words do not imply identical history-aware text features.",
             ha="center", fontsize=6.5, color=INK)
    export(fig, out, "01_same_words_scenes")


def box(ax, x, y, width, height, text, color=INK, fill="#F6F7F8", size=8):
    patch = FancyBboxPatch((x, y), width, height,
                          boxstyle="round,pad=0.03,rounding_size=0.10",
                          edgecolor=color, facecolor=fill, linewidth=.8)
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
            fontsize=size, color=INK)


def arrow(ax, start, end, color=INK):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8,
                                linewidth=.8, color=color))


def amm_figure(out: Path):
    fig = plt.figure(figsize=(180 / 25.4, 120 / 25.4))
    ax = fig.add_axes([.015, .015, .97, .97])
    ax.set(xlim=(0, 13), ylim=(0, 9))
    ax.set_axis_off()
    ax.text(6.5, 8.60, "AMM: learned projection views and axis-wise interaction",
            ha="center", fontsize=10, fontweight="bold", color=INK)
    ax.text(.35, 8.10, "a   Expand each branch representation", fontsize=8,
            fontweight="bold", color=INK)
    ax.text(1.28, 7.52, "After preceding MCA", ha="center", fontsize=7, color=INK)
    for m, y in zip(("t", "a", "v"), (6.72, 5.86, 5.00)):
        box(ax, .42, y, 1.7, .58, rf"$h_{m}\in\mathbb{{R}}^{{256}}$", color=BLUE)
        arrow(ax, (2.17, y + .29), (2.77, y + .29))
    box(ax, 2.82, 4.92, 3.05, 2.43,
        "6 learned projections\n(shared across branches)\n\n"
        r"$z_{m,s}=W_s h_m+b_s$", color=BLUE, fill="#EFF4FA", size=7.5)
    for y in (7.01, 6.15, 5.29):
        arrow(ax, (5.93, y), (6.30, y))
    ax.text(9.72, 7.53, "Projection axis S = 6", ha="center", fontsize=7.5, color=BLUE)
    for row, (m, y) in enumerate(zip(("t", "a", "v"), (6.72, 5.86, 5.00))):
        ax.text(6.60, y + .29, m.upper(), va="center", ha="right", fontsize=7, color=INK)
        for s in range(1, 7):
            x = 6.78 + (s - 1) * .94
            box(ax, x, y, .82, .58, rf"$z_{{{m},{s}}}$",
                color=BLUE, fill="#EFF4FA", size=7)
    ax.text(9.55, 4.55, r"$Z\in\mathbb{R}^{3\times6\times256}$  (M, S, D)",
            ha="center", fontsize=8, color=INK)

    ax.text(.35, 4.05, "b   Mix axes, then aggregate", fontsize=8,
            fontweight="bold", color=INK)
    for x, label, order in [(0.52, "Block 1", ("S", "M", "D")),
                            (4.32, "Block 2", ("M", "S", "D"))]:
        box(ax, x, 2.42, 3.12, 1.22, "", fill="white")
        ax.text(x + 1.56, 3.36, label, ha="center", fontsize=8, color=INK)
        for j, axis in enumerate(order):
            tint = {"S": BLUE, "M": TEAL, "D": VIOLET}[axis]
            tile_x = x + .20 + j * .96
            box(ax, tile_x, 2.64, .72, .40, axis, color=tint, fill="white", size=8)
            if j < 2:
                arrow(ax, (tile_x + .76, 2.84), (tile_x + .92, 2.84))
    arrow(ax, (3.71, 3.02), (4.23, 3.02))
    arrow(ax, (7.51, 3.02), (7.95, 3.02))
    box(ax, 8.02, 2.42, 1.65, 1.22, "Mean\nacross S", size=8)
    arrow(ax, (9.74, 3.02), (10.19, 3.02))
    box(ax, 10.25, 2.42, 2.13, 1.22,
        "3 branch vectors\n256 dimensions each\nPooling + pair residual", size=6.7)

    for x, title, detail, color in [
        (.55, "S: projection mixing", "Across views within a branch", BLUE),
        (4.80, "M: modality mixing", "Across branches at each (s, d)", TEAL),
        (9.00, "D: channel mixing", "Within each (m, s) vector", VIOLET),
    ]:
        ax.text(x, 1.76, title, fontsize=7.5, fontweight="bold", color=color)
        ax.text(x, 1.33, detail, fontsize=6.7, color=INK)
    ax.text(6.5, .76, "M uses one learned 3-by-3 routing matrix shared across S and D.",
            ha="center", fontsize=7, color=INK)
    ax.text(6.5, .32,
            "S is a learned projection axis. Semantic roles of individual views are not established.",
            ha="center", fontsize=6.7, color=INK)
    export(fig, out, "02_amm_projection_axes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path,
                        default=Path("/data2/yb/OpenDataLab___MELD/raw/MELD/MELD.Raw"))
    parser.add_argument("--audio-root", type=Path,
                        default=Path("/data2/yb/multimodalERC/MELD/Dataset/Data/local_meld"))
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    images, waveforms, records = load_cases(args.raw_root, args.audio_root, args.out)
    pd.DataFrame(records).to_csv(args.out / "source_data.csv", index=False)
    scene_figure(images, waveforms, records, args.out)
    amm_figure(args.out)
    print(json.dumps({"output": str(args.out), "verified_cases": len(records),
                      "figures": ["01_same_words_scenes", "02_amm_projection_axes"]}))


if __name__ == "__main__":
    main()
