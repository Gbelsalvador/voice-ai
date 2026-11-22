from __future__ import annotations

import argparse
import csv
from pathlib import Path


def convert(root: Path, split: str, output: Path, speaker: str | None, limit: int) -> int:
    source_path = root / f"{split}.tsv"
    if not source_path.is_file():
        raise FileNotFoundError(f"TSV introuvable : {source_path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    count = missing = 0
    with source_path.open("r", encoding="utf-8-sig", newline="") as source, output.open("w", encoding="utf-8", newline="") as dest:
        reader = csv.DictReader(source, delimiter="\t")
        if not reader.fieldnames or not {"path", "sentence"}.issubset(reader.fieldnames):
            raise ValueError(f"{source_path} doit contenir les colonnes path et sentence")
        writer = csv.DictWriter(dest, fieldnames=["audio", "text"])
        writer.writeheader()
        for row in reader:
            if speaker and row.get("client_id") != speaker:
                continue
            audio = root / "clips" / (row.get("path") or "")
            text = (row.get("sentence") or "").strip()
            if not audio.is_file():
                missing += 1
                continue
            if text:
                writer.writerow({"audio": str(audio.resolve()), "text": text})
                count += 1
                if limit and count >= limit:
                    break
    print(f"{split}: {count} exemples -> {output.resolve()} ({missing} audio absents)")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Crée les manifestes train/dev/test Common Voice.")
    parser.add_argument("--root", required=True, help="Dossier Common Voice contenant train.tsv et clips/")
    parser.add_argument("--output-dir", default="manifests/commonvoice")
    parser.add_argument("--speaker", help="Filtrer un client_id (même filtre sur les trois splits)")
    parser.add_argument("--limit", type=int, default=0, help="Maximum d'exemples par split; 0 = tous")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit doit être positif ou nul")
    root, out = Path(args.root).resolve(), Path(args.output_dir)
    for split in ("train", "dev", "test"):
        convert(root, split, out / f"{split}.csv", args.speaker, args.limit)


if __name__ == "__main__":
    main()
