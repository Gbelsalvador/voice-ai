from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from common import SpeechDataset, build_vocab, read_manifest, save_vocab, set_seed, split_rows, write_json
from stt_model import STTModel


def collate(batch):
    feats, targets, _ = zip(*batch)
    x_lengths = torch.tensor([len(x) for x in feats], dtype=torch.long)
    y_lengths = torch.tensor([len(y) for y in targets], dtype=torch.long)
    max_t = int(x_lengths.max())
    x = torch.zeros(len(feats), max_t, feats[0].shape[1], dtype=torch.float32)
    for i, feat in enumerate(feats):
        x[i, :len(feat)] = feat
    y = torch.cat(targets)
    return x, x_lengths, y, y_lengths


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    total_batches = 0
    for x, x_len, y, y_len in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            logits, out_len = model(x, x_len.to(device))
            log_probs = logits.log_softmax(-1).transpose(0, 1)
            loss = criterion(log_probs, y, out_len.cpu(), y_len)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        total_loss += float(loss.detach().cpu())
        total_batches += 1
    return total_loss / max(total_batches, 1)


def main():
    p = argparse.ArgumentParser(description="Entraîner un STT CTC depuis une initialisation aléatoire.")
    p.add_argument("--manifest", required=True, help="CSV avec colonnes audio,text")
    p.add_argument("--out", default="checkpoints/stt")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = read_manifest(args.manifest)
    train_rows, val_rows = split_rows(rows, seed=args.seed)
    vocab = build_vocab((r["text"] for r in train_rows), "stt")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_vocab(vocab, out / "vocab.json")
    train_ds = SpeechDataset(train_rows, vocab, "stt")
    val_ds = SpeechDataset(val_rows, vocab, "stt")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate)

    model = STTModel(len(vocab)).to(device)
    criterion = nn.CTCLoss(blank=vocab["<blank>"], zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    best_val = float("inf")
    history = []
    print(f"Device={device} | train={len(train_ds)} | val={len(val_ds)} | vocab={len(vocab)}")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_dl, criterion, optimizer, device, True)
        val_loss = run_epoch(model, val_dl, criterion, None, device, False)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:03d}/{args.epochs} | train CTC={train_loss:.4f} | val CTC={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "vocab": vocab, "n_mels": 80}, out / "best.pt")
        write_json({"history": history, "best_val_loss": best_val}, out / "history.json")

    print(f"Meilleur checkpoint : {out / 'best.pt'}")


if __name__ == "__main__":
    main()
