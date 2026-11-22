from __future__ import annotations

import argparse

import librosa
import numpy as np
import torch

from common import load_vocab
from stt_model import STTModel


def extract(audio_path: str):
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=400, hop_length=160, win_length=400, n_mels=80, fmin=40, fmax=8000, power=2.0)
    x = np.log(np.maximum(mel, 1e-5)).T.astype(np.float32)
    x = (x - x.mean()) / (x.std() + 1e-5)
    return torch.from_numpy(x).unsqueeze(0)


def greedy_decode(logits, vocab):
    id_to_token = {v: k for k, v in vocab.items()}
    ids = logits.argmax(-1)[0].tolist()
    blank = vocab["<blank>"]
    previous = None
    chars = []
    for idx in ids:
        if idx != previous and idx != blank and id_to_token.get(idx) not in ("<unk>", "<blank>"):
            chars.append(id_to_token[idx])
        previous = idx
    return "".join(chars).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--audio", required=True)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    vocab = ckpt["vocab"]
    model = STTModel(len(vocab), n_mels=ckpt.get("n_mels", 80)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    x = extract(args.audio).to(device)
    with torch.no_grad():
        logits, _ = model(x, torch.tensor([x.shape[1]], device=device))
    print(greedy_decode(logits, vocab))


if __name__ == "__main__":
    main()
