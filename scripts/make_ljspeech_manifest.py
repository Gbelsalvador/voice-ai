from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Convertit metadata.csv de LJSpeech (id|texte|texte_normalise) en manifest audio,text.")
    p.add_argument("--root", required=True, help="Dossier racine LJSpeech contenant metadata.csv et wavs/")
    p.add_argument("--output", default="manifests/tts.csv")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    root = Path(args.root).resolve()
    metadata = root / "metadata.csv"
    if not metadata.exists():
        raise FileNotFoundError(metadata)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with metadata.open("r", encoding="utf-8", newline="") as source, output.open("w", encoding="utf-8", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=["audio", "text"])
        writer.writeheader()
        for row in csv.reader(source, delimiter="|"):
            if len(row) < 2:
                continue
            audio = root / "wavs" / f"{row[0]}.wav"
            text = row[2] if len(row) >= 3 and row[2].strip() else row[1]
            if audio.is_file() and text.strip():
                writer.writerow({"audio": str(audio), "text": text.strip()})
                count += 1
                if args.limit and count >= args.limit:
                    break
    print(f"Manifest créé : {output.resolve()} ({count} lignes)")


if __name__ == "__main__":
    main()
