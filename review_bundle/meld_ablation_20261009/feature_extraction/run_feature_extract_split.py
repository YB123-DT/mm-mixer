import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Model'))
from feature_extract import train_meld_text_encoder, extract_text_features, extract_visual_features, extract_audio_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "dev", "test"], required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    meld_raw_dir = os.environ.get("MELD_RAW_DIR", "/data2/yb/OpenDataLab___MELD/raw/MELD/MELD.Raw")
    meld_work_dir = os.environ.get("MELD_WORK_DIR", "/data2/yb/multimodalERC/MELD/Dataset/Data/local_meld")
    feature_root = os.environ.get("FEATURE_ROOT", "/data2/yb/multimodalERC/MELD/Dataset/Data")
    lipsync_model_path = os.environ.get("MELD_LIPSYNC_MODEL_PATH", os.path.join(feature_root, "lipsync_model_meld.pth"))
    finetuned_text_dir = os.environ.get("FINETUNED_TEXT_DIR", os.path.join(feature_root, "fine_tuned_model_meld"))

    csv_path = os.path.join(meld_raw_dir, f"{args.split}_sent_emo.csv")
    video_dir = os.path.join(meld_work_dir, f"{args.split}_video")
    audio_dir = os.path.join(meld_work_dir, f"{args.split}_audio")
    out_dir = os.path.join(feature_root, f"{args.split}_features")

    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    if not os.path.isdir(video_dir):
        raise FileNotFoundError(video_dir)
    if not os.path.isdir(audio_dir):
        raise FileNotFoundError(audio_dir)
    if not os.path.exists(lipsync_model_path):
        raise FileNotFoundError(lipsync_model_path)

    if args.split == "train" and not os.path.exists(finetuned_text_dir):
        train_meld_text_encoder(
            input_csv=csv_path,
            model_name="roberta-large",
            output_model_dir=finetuned_text_dir,
            batch_size=4,
            num_epochs=10,
            learning_rate=1e-6,
        )

    if not os.path.exists(finetuned_text_dir):
        raise FileNotFoundError(
            f"Missing text encoder dir: {finetuned_text_dir}. Run train split first or ensure it exists."
        )

    extract_text_features(csv_path, os.path.join(out_dir, "text_features.json"), finetuned_text_dir, batch_size=16)
    extract_visual_features(video_dir, audio_dir, lipsync_model_path, os.path.join(out_dir, "visual_features.json"))
    extract_audio_features(
        audio_dir,
        os.path.join(out_dir, "audio_features.json"),
        model_name="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
        sample_rate=16000,
    )
    print(f"{args.split} feature extraction complete.")


if __name__ == "__main__":
    main()
