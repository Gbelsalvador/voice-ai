from __future__ import annotations

import argparse

import librosa
import numpy as np
import torch

from common import load_vocab, normalize_text, read_manifest
from stt_model import STTModel


def extract(audio_path: str):
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=400, hop_length=160, win_length=400, n_mels=80, fmin=40, fmax=8000, power=2.0)
    x = np.log(np.maximum(mel, 1e-5)).T.astype(np.float32)
    x = (x - x.mean()) / (x.std() + 1e-5)
    return torch.from_numpy(x).unsqueeze(0)


def decode(logits, vocab):
    inverse = {v: k for k, v in vocab.items()}
    blank = vocab["<blank>"]
    output, previous = [], None
    for idx in logits.argmax(-1)[0].tolist():
        if idx != previous and idx != blank:
            token = inverse.get(idx, "")
            if token not in ("<unk>", "<blank>"):
                output.append(token)
        previous = idx
    return "".join(output).strip()


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def main():
    p = argparse.ArgumentParser(description="Calcule WER et CER sur un manifest tenu à part du train.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    vocab = ckpt["vocab"]
    model = STTModel(len(vocab), n_mels=ckpt.get("n_mels", 80)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    total_word_errors = total_words = total_char_errors = total_chars = 0
    rows = read_manifest(args.manifest)
    with torch.no_grad():
        for row in rows:
            x = extract(row["audio"]).to(device)
            logits, _ = model(x, torch.tensor([x.shape[1]], device=device))
            hyp = normalize_text(decode(logits, vocab))
            ref = normalize_text(row["text"])
            ref_words, hyp_words = ref.split(), hyp.split()
            total_word_errors += edit_distance(ref_words, hyp_words)
            total_words += max(len(ref_words), 1)
            total_char_errors += edit_distance(list(ref.replace(" ", "")), list(hyp.replace(" ", "")))
            total_chars += max(len(ref.replace(" ", "")), 1)
    print(f"Échantillons : {len(rows)}")
    print(f"WER : {100 * total_word_errors / total_words:.2f}%")
    print(f"CER : {100 * total_char_errors / total_chars:.2f}%")
    print("Une valeur plus basse est meilleure. Calculer ces métriques sur un jeu test jamais utilisé pendant l'entraînement.")


if __name__ == "__main__":
    main()
