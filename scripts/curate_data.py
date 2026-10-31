"""Clean raw pretraining text: drop low-quality documents and (near-)duplicates.

Reads ``.jsonl`` or ``.jsonl.zst`` files with a ``text`` field (the format of The Pile and
of ``scripts/data_download.py``) and writes the survivors, in the same format, to
``--out_dir/curated.jsonl.zst``. Point ``scripts/data_preprocess.py`` at that folder to
tokenize it.

    python scripts/curate_data.py --input data/raw/train --out_dir data/curated/train
    python scripts/curate_data.py --input data/raw/train --limit 20000 --dry_run   # stats only
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)  # run from the repo without installing

import argparse
import io
import json
import os
from collections import Counter
from collections.abc import Iterator

import zstandard as zstd
from tqdm import tqdm

from src.curation import Deduplicator, QualityConfig, quality_issues


def input_files(path: str) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    files = sorted([*p.glob("*.jsonl"), *p.glob("*.jsonl.zst")])
    if not files:
        raise SystemExit(f"no .jsonl or .jsonl.zst files in {path}")
    return files


def iter_documents(files: list[Path]) -> Iterator[dict]:
    for f in files:
        if f.suffix == ".zst":
            fh = zstd.ZstdDecompressor().stream_reader(open(f, "rb"))
            lines = io.TextIOWrapper(fh, encoding="utf-8")
        else:
            lines = open(f, encoding="utf-8")
        with lines:
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(doc, dict) and isinstance(doc.get("text"), str):
                    yield doc


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--input", required=True, help="a .jsonl(.zst) file or a folder of them"
    )
    p.add_argument("--out_dir", default="data/curated")
    p.add_argument(
        "--limit", type=int, default=0, help="stop after this many documents"
    )
    p.add_argument("--min_words", type=int, default=QualityConfig.min_words)
    p.add_argument("--no_quality", action="store_true", help="skip the quality filters")
    p.add_argument("--no_dedup", action="store_true", help="skip duplicate removal")
    p.add_argument(
        "--threshold", type=float, default=0.8, help="near-duplicate Jaccard"
    )
    p.add_argument("--num_perm", type=int, default=128)
    p.add_argument("--bands", type=int, default=16)
    p.add_argument(
        "--dry_run", action="store_true", help="report statistics, write nothing"
    )
    p.add_argument(
        "--rejects", default=None, help="also write dropped docs (with reasons) here"
    )
    args = p.parse_args()

    qcfg = QualityConfig(min_words=args.min_words)
    dedup = (
        None
        if args.no_dedup
        else Deduplicator(args.threshold, args.num_perm, args.bands)
    )
    reasons: Counter[str] = Counter()
    n_in = n_out = chars_in = chars_out = 0

    out = rejects = None
    if not args.dry_run:
        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(args.out_dir, "curated.jsonl.zst")
        out = zstd.ZstdCompressor(level=3).stream_writer(open(out_path, "wb"))
    if args.rejects:
        rejects = open(args.rejects, "w", encoding="utf-8")

    for doc in tqdm(
        iter_documents(input_files(args.input)), desc="curating", unit="doc"
    ):
        if args.limit and n_in >= args.limit:
            break
        text = doc["text"]
        n_in += 1
        chars_in += len(text)
        issues = [] if args.no_quality else quality_issues(text, qcfg)
        if not issues and dedup is not None and dedup.is_duplicate(text):
            issues = ["duplicate"]
        if issues:
            reasons.update(issues)
            if rejects:
                rejects.write(
                    json.dumps({"reasons": issues, "text": text[:2000]}) + "\n"
                )
            continue
        n_out += 1
        chars_out += len(text)
        if out is not None:
            out.write((json.dumps(doc, ensure_ascii=False) + "\n").encode("utf-8"))

    if out is not None:
        out.close()
    if rejects:
        rejects.close()

    print(f"\n{'documents in':<24}{n_in:>12,}")
    if reasons:
        print("  rules that fired (one document can break several):")
    for reason, count in reasons.most_common():
        print(f"  {reason:<22}{count:>12,}  ({100 * count / max(1, n_in):.1f}%)")
    if dedup is not None:
        print(f"  {'(exact duplicates)':<22}{dedup.n_exact:>12,}")
        print(f"  {'(near duplicates)':<22}{dedup.n_near:>12,}")
    print(f"{'documents kept':<24}{n_out:>12,}  ({100 * n_out / max(1, n_in):.1f}%)")
    print(f"{'characters kept':<24}{100 * chars_out / max(1, chars_in):>11.1f}%")
    if not args.dry_run:
        print(f"-> {os.path.join(args.out_dir, 'curated.jsonl.zst')}")


if __name__ == "__main__":
    main()
