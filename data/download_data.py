"""
Download Amazon Reviews 2023 (McAuley-Lab) subsets for Electronics and All_Beauty.

Source: HuggingFace McAuley-Lab/Amazon-Reviews-2023 raw JSONL files.
- All_Beauty (~330MB): downloaded fully.
- Electronics (22GB raw): streamed, first ELECTRONICS_SAMPLE_ROWS rows saved.

Outputs: data/raw/All_Beauty.jsonl.gz, data/raw/Electronics.jsonl.gz
Idempotent: skips download if output file already exists.
"""

import gzip
import json
import os
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"

HF_BASE = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories"

# Electronics is 22GB raw; stream enough rows to ensure 15K/class after 80/10/10 split.
# Amazon reviews are ~85% positive; neutral/negative are ~6% each.
# To get 15K negatives in train (80% of total), need ~15000/0.06/0.8 = ~312K raw rows.
# We stream 320K to have comfortable margin.
ELECTRONICS_SAMPLE_ROWS = 320_000
ALL_BEAUTY_SAMPLE_ROWS = 40_000  # full file is ~330MB; 40K rows sufficient for supplement

DATASETS = [
    {
        "name": "All_Beauty",
        "url": f"{HF_BASE}/All_Beauty.jsonl",
        "output": RAW_DIR / "All_Beauty.jsonl.gz",
        "max_rows": ALL_BEAUTY_SAMPLE_ROWS,
    },
    {
        "name": "Electronics",
        "url": f"{HF_BASE}/Electronics.jsonl",
        "output": RAW_DIR / "Electronics.jsonl.gz",
        "max_rows": ELECTRONICS_SAMPLE_ROWS,
    },
]


def stream_and_save(name: str, url: str, output_path: Path, max_rows: int) -> int:
    """Stream JSONL from url, write first max_rows valid records to gzipped output."""
    print(f"[{name}] Streaming from {url}")
    print(f"[{name}] Saving up to {max_rows:,} rows -> {output_path}")

    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Mozilla/5.0 (academic research)")

    rows_written = 0
    rows_skipped = 0

    with urllib.request.urlopen(req, timeout=120) as response, \
         gzip.open(output_path, "wt", encoding="utf-8") as out_file:

        for raw_line in response:
            if rows_written >= max_rows:
                break

            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                rows_skipped += 1
                continue

            # Keep only records with required fields
            if not record.get("text") or record.get("rating") is None:
                rows_skipped += 1
                continue

            out_file.write(json.dumps(record) + "\n")
            rows_written += 1

            if rows_written % 10_000 == 0:
                print(f"[{name}]   {rows_written:,} rows written...")

    print(f"[{name}] Done: {rows_written:,} rows written, {rows_skipped} skipped.")
    return rows_written


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        name = dataset["name"]
        output_path = dataset["output"]

        if output_path.exists():
            size_mb = output_path.stat().st_size / 1e6
            print(f"[{name}] Already exists ({size_mb:.1f} MB) — skipping download.")
            continue

        rows = stream_and_save(
            name=name,
            url=dataset["url"],
            output_path=output_path,
            max_rows=dataset["max_rows"],
        )

        size_mb = output_path.stat().st_size / 1e6
        print(f"[{name}] Saved {rows:,} rows, {size_mb:.1f} MB compressed.")

    print("\nDownload complete. Files in data/raw/:")
    for f in sorted(RAW_DIR.glob("*.jsonl.gz")):
        print(f"  {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
