from __future__ import annotations

import io
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

try:  # Supporte `uvicorn src.api:app` et `python src/api.py`.
    from .stt_model import STTModel
except ImportError:  # pragma: no cover - chemin d'exécution directe
    from stt_model import STTModel


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "stt_tshiluba" / "best.pt"
CHECKPOINT_PATH = Path(os.getenv("STT_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
if not CHECKPOINT_PATH.is_absolute():
    CHECKPOINT_PATH = ROOT / CHECKPOINT_PATH

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = FastAPI(title="Voice AI API", version="1.0.0")
WEB_PAGE = Path(__file__).resolve().parent / "static" / "index.html"


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str


class TranscriptionResponse(BaseModel):
    text: str


@app.get("/", include_in_schema=False)
def web_interface():
    return FileResponse(WEB_PAGE)


def load_model():
    if not CHECKPOINT_PATH.is_file():
        raise RuntimeError(f"Checkpoint introuvable : {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    vocab = checkpoint["vocab"]
    model = STTModel(len(vocab), n_mels=checkpoint.get("n_mels", 80)).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, vocab


try:
    model, vocab = load_model()
    load_error = None
except Exception as exc:
    model, vocab = None, None
    load_error = str(exc)


def decode(logits: torch.Tensor, token_vocab: dict[str, int]) -> str:
    id_to_token = {value: key for key, value in token_vocab.items()}
    blank = token_vocab["<blank>"]
    previous = None
    chars = []
    for idx in logits.argmax(-1)[0].tolist():
        if idx != previous and idx != blank and id_to_token.get(idx) not in ("<unk>", "<blank>"):
            chars.append(id_to_token[idx])
        previous = idx
    return "".join(chars).strip()


@app.get("/health", response_model=HealthResponse)
def health():
    if model is None:
        raise HTTPException(status_code=503, detail=load_error)
    return HealthResponse(status="ok", model=str(CHECKPOINT_PATH), device=str(DEVICE))


@app.post("/stt/transcribe", response_model=TranscriptionResponse)
async def transcribe(audio: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail=load_error)
    try:
        content = await audio.read()
        if not content:
            raise ValueError("Le fichier audio est vide.")
        waveform, sample_rate = sf.read(io.BytesIO(content), dtype="float32", always_2d=True)
        waveform = waveform.mean(axis=1)
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
        mel = librosa.feature.melspectrogram(
            y=waveform, sr=16000, n_fft=400, hop_length=160, win_length=400,
            n_mels=80, fmin=40, fmax=8000, power=2.0,
        )
        features = np.log(np.maximum(mel, 1e-5)).T.astype(np.float32)
        features = (features - features.mean()) / (features.std() + 1e-5)
        x = torch.from_numpy(features).unsqueeze(0).to(DEVICE)
        lengths = torch.tensor([x.shape[1]], device=DEVICE)
        with torch.inference_mode():
            logits, _ = model(x, lengths)
        return TranscriptionResponse(text=decode(logits, vocab))
    except (ValueError, RuntimeError, sf.SoundFileError) as exc:
        raise HTTPException(status_code=400, detail=f"Audio invalide ou non pris en charge : {exc}") from exc
