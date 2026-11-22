from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_text(text: str) -> str:
    """Normalisation légère : conserve les accents et la ponctuation utile."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def read_manifest(csv_path: str | Path) -> list[dict[str, str]]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest introuvable : {path}")
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"audio", "text"}.issubset(reader.fieldnames):
            raise ValueError("Le CSV doit contenir exactement au minimum les colonnes 'audio' et 'text'.")
        for i, row in enumerate(reader, start=2):
            audio = str(row["audio"]).strip()
            text = normalize_text(str(row["text"] or ""))
            if not audio or not text:
                continue
            audio_path = Path(audio)
            if not audio_path.is_absolute():
                audio_path = (path.parent / audio_path).resolve()
            if not audio_path.is_file():
                raise FileNotFoundError(f"Audio absent à la ligne {i} : {audio_path}")
            rows.append({"audio": str(audio_path), "text": text})
    if len(rows) < 2:
        raise ValueError("Il faut au moins deux exemples audio/texte dans le manifest.")
    return rows


def split_rows(rows: list[dict[str, str]], val_fraction: float = 0.05, seed: int = 42):
    if len(rows) < 10:
        val_fraction = min(val_fraction, 0.2)
    shuffled = rows.copy()
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]


def build_vocab(texts: Iterable[str], kind: str) -> dict:
    chars = sorted({c for t in texts for c in normalize_text(t)})
    if kind == "stt":
        tokens = ["<blank>", "<unk>"] + chars
    elif kind == "tts":
        tokens = ["<pad>", "<unk>", "<eos>"] + chars
    else:
        raise ValueError("kind doit être 'stt' ou 'tts'")
    return {token: i for i, token in enumerate(tokens)}


def save_vocab(vocab: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_vocab(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def encode_text(text: str, vocab: dict, kind: str) -> list[int]:
    unk = vocab["<unk>"]
    ids = [vocab.get(c, unk) for c in normalize_text(text)]
    if kind == "tts":
        ids.append(vocab["<eos>"])
    return ids


def _cache_path(audio_path: str, cache_dir: str | Path, tag: str) -> Path:
    digest = hashlib.sha1(str(Path(audio_path).resolve()).encode("utf-8")).hexdigest()
    folder = Path(cache_dir) / tag
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{digest}.npy"


def audio_to_features(
    audio_path: str,
    task: str,
    cache_dir: str | Path,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int = 80,
) -> np.ndarray:
    """Extrait et met en cache des log-mels. STT : normalisés par énoncé; TTS : log-mels non normalisés."""
    tag = f"{task}_{sample_rate}_{n_fft}_{hop_length}_{win_length}_{n_mels}"
    cache_file = _cache_path(audio_path, cache_dir, tag)
    if cache_file.exists():
        return np.load(cache_file)

    waveform, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    if waveform.size == 0:
        raise ValueError(f"Audio vide : {audio_path}")
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=40,
        fmax=sample_rate / 2,
        power=2.0,
    )
    log_mel = np.log(np.maximum(mel, 1e-5)).T.astype(np.float32)
    if task == "stt":
        mean = float(log_mel.mean())
        std = float(log_mel.std()) + 1e-5
        log_mel = (log_mel - mean) / std
    np.save(cache_file, log_mel)
    return log_mel


class SpeechDataset(Dataset):
    def __init__(self, rows, vocab, task, cache_dir="data/cache"):
        self.rows = rows
        self.vocab = vocab
        self.task = task
        self.cache_dir = cache_dir
        if task == "stt":
            self.audio_cfg = dict(sample_rate=16000, n_fft=400, hop_length=160, win_length=400)
        else:
            self.audio_cfg = dict(sample_rate=22050, n_fft=1024, hop_length=256, win_length=1024)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        feat = audio_to_features(
            row["audio"], self.task, self.cache_dir, **self.audio_cfg
        )
        ids = encode_text(row["text"], self.vocab, self.task)
        return torch.from_numpy(feat), torch.tensor(ids, dtype=torch.long), row["text"]


def write_json(data: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
