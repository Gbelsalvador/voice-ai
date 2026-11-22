from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Convertit validated.tsv de Common Voice en manifest audio,text.")
    p.add_argument("--root", required=True, help="Dossier racine Common Voice de la langue")
    p.add_argument("--tsv", default="validated.tsv", help="Fichier TSV, chemin relatif à root si besoin")
    p.add_argument("--output", default="manifests/stt.csv")
    p.add_argument("--speaker", default=None, help="Filtrer sur client_id pour un locuteur donné")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    root = Path(args.root).resolve()
    tsv = Path(args.tsv)
    if not tsv.is_absolute():
        tsv = root / tsv
    if not tsv.exists():
        raise FileNotFoundError(tsv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with tsv.open("r", encoding="utf-8", newline="") as source, output.open("w", encoding="utf-8", newline="") as dest:
        reader = csv.DictReader(source, delimiter="\t")
        writer = csv.DictWriter(dest, fieldnames=["audio", "text"])
        writer.writeheader()
        for row in reader:
            if args.speaker and row.get("client_id") != args.speaker:
                continue
            audio = root / "clips" / row["path"]
            text = (row.get("sentence") or "").strip()
            if audio.is_file() and text:
                writer.writerow({"audio": str(audio), "text": text})
                count += 1
                if args.limit and count >= args.limit:
                    break
    print(f"Manifest créé : {output.resolve()} ({count} lignes)")


if __name__ == "__main__":
    main()
