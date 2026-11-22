from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import SpeechDataset, build_vocab, read_manifest, save_vocab, set_seed, split_rows, write_json
from tts_model import TTSModel


def collate(batch):
    mels, texts, _ = zip(*batch)
    mel_lengths = torch.tensor([len(x) for x in mels], dtype=torch.long)
    text_lengths = torch.tensor([len(x) for x in texts], dtype=torch.long)
    max_mel = int(mel_lengths.max())
    max_text = int(text_lengths.max())
    mel_batch = torch.zeros(len(mels), max_mel, mels[0].shape[1], dtype=torch.float32)
    text_batch = torch.zeros(len(texts), max_text, dtype=torch.long)
    for i, (mel, text) in enumerate(zip(mels, texts)):
        mel_batch[i, :len(mel)] = mel
        text_batch[i, :len(text)] = text
    return text_batch, text_lengths, mel_batch, mel_lengths


def run_epoch(model, loader, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    batches = 0
    for text, text_len, target, mel_len in loader:
        text = text.to(device)
        target = target.to(device)
        text_len = text_len.to(device)
        mel_len = mel_len.to(device)
        with torch.set_grad_enabled(train):
            pred_mel, pred_stop, _ = model(text, text_len, target)
            steps = target.size(1)
            mask = torch.arange(steps, device=device).unsqueeze(0) < mel_len.unsqueeze(1)
            mel_error = torch.abs(pred_mel - target) * mask.unsqueeze(-1)
            mel_loss = mel_error.sum() / (mask.sum().clamp(min=1) * target.size(-1))
            stop_target = torch.zeros_like(pred_stop)
            stop_target[torch.arange(len(mel_len), device=device), mel_len - 1] = 1.0
            stop_error = F.binary_cross_entropy_with_logits(pred_stop, stop_target, reduction="none")
            stop_loss = (stop_error * mask).sum() / mask.sum().clamp(min=1)
            loss = mel_loss + 0.2 * stop_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total_loss += float(loss.detach().cpu())
        batches += 1
    return total_loss / max(batches, 1)


def main():
    p = argparse.ArgumentParser(description="Entraîner un TTS texte->mel depuis une initialisation aléatoire.")
    p.add_argument("--manifest", required=True, help="CSV avec colonnes audio,text")
    p.add_argument("--out", default="checkpoints/tts")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = read_manifest(args.manifest)
    train_rows, val_rows = split_rows(rows, val_fraction=0.05, seed=args.seed)
    vocab = build_vocab((r["text"] for r in train_rows), "tts")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_vocab(vocab, out / "vocab.json")
    train_ds = SpeechDataset(train_rows, vocab, "tts")
    val_ds = SpeechDataset(val_rows, vocab, "tts")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate)

    model = TTSModel(len(vocab), n_mels=80).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    best_val = float("inf")
    history = []
    print(f"Device={device} | train={len(train_ds)} | val={len(val_ds)} | vocab={len(vocab)}")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_dl, optimizer, device, True)
        val_loss = run_epoch(model, val_dl, optimizer, device, False)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:03d}/{args.epochs} | train={train_loss:.4f} | val={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model": model.state_dict(),
                "vocab": vocab,
                "n_mels": 80,
                "sample_rate": 22050,
                "n_fft": 1024,
                "hop_length": 256,
                "win_length": 1024,
            }, out / "best.pt")
        write_json({"history": history, "best_val_loss": best_val}, out / "history.json")
    print(f"Meilleur checkpoint : {out / 'best.pt'}")


if __name__ == "__main__":
    main()
