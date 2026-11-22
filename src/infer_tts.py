from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from common import encode_text
from tts_model import TTSModel


def main():
    p = argparse.ArgumentParser(description="Synthèse vocale avec le TTS entraîné.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--output", default="generated.wav")
    p.add_argument("--max-frames", type=int, default=1200)
    p.add_argument("--griffin-lim-iterations", type=int, default=64)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    vocab = ckpt["vocab"]
    ids = encode_text(args.text, vocab, "tts")
    text = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    text_len = torch.tensor([len(ids)], dtype=torch.long, device=device)

    model = TTSModel(len(vocab), n_mels=ckpt.get("n_mels", 80)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    with torch.no_grad():
        mel_log, _ = model.generate(text, text_len, max_steps=args.max_frames)
    mel_power = np.exp(np.clip(mel_log[0].cpu().numpy(), -15.0, 5.0)).T
    mel_power = np.maximum(mel_power, 1e-8)

    # Griffin-Lim : algorithme d'inversion, pas un vocoder neuronal entraîné.
    waveform = librosa.feature.inverse.mel_to_audio(
        mel_power,
        sr=ckpt.get("sample_rate", 22050),
        n_fft=ckpt.get("n_fft", 1024),
        hop_length=ckpt.get("hop_length", 256),
        win_length=ckpt.get("win_length", 1024),
        power=2.0,
        fmin=40,
        fmax=ckpt.get("sample_rate", 22050) / 2,
        n_iter=args.griffin_lim_iterations,
    )
    waveform = np.nan_to_num(waveform)
    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform = 0.95 * waveform / peak
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, waveform, ckpt.get("sample_rate", 22050))
    print(f"Audio généré : {output.resolve()}")


if __name__ == "__main__":
    main()
