from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np


def main():
    p = argparse.ArgumentParser(description="Contrôle les fichiers d'un manifest audio,text.")
    p.add_argument("manifest")
    p.add_argument("--sample", type=int, default=0, help="Analyser au maximum N fichiers; 0 = tous")
    args = p.parse_args()
    import csv
    rows = []
    with Path(args.manifest).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"audio", "text"}.issubset(rows[0]):
        raise ValueError("CSV vide ou colonnes audio,text absentes")
    subset = rows[:args.sample] if args.sample else rows
    durations, missing, empty = [], [], 0
    for row in subset:
        path = Path(row["audio"])
        if not path.is_absolute():
            path = (Path(args.manifest).resolve().parent / path).resolve()
        if not path.is_file():
            missing.append(str(path))
            continue
        if not row.get("text", "").strip():
            empty += 1
            continue
        try:
            info = librosa.get_duration(path=str(path))
            durations.append(info)
        except Exception as exc:
            print(f"Audio illisible : {path} ({exc})")
    print(f"Exemples examinés : {len(subset)}")
    print(f"Fichiers manquants : {len(missing)} | textes vides : {empty}")
    if durations:
        print(f"Durée totale : {sum(durations)/3600:.2f} h")
        print(f"Durée moyenne : {np.mean(durations):.2f} s | médiane : {np.median(durations):.2f} s")
    if missing:
        print("Premiers chemins absents :")
        print("\n".join(missing[:10]))


if __name__ == "__main__":
    main()
