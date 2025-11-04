#!/usr/bin/env python3
"""
Batch CSV Clipper (retargeted trajectory CSVs)

- Reads clip specs from a CSV: columns = file,start,end[,out]
- Clips ROWS [start..end] (1-indexed, inclusive) from each CSV
- Writes results to out_dir, using either 'out' or auto-named '*_seg{k}.csv'
- Keeps headers if present; otherwise treats data as headerless
- Does not change delimiters (','), column counts, or dtypes

Usage:
  python clip_csv_batch.py \
    --csv segments_sangwoo_251012.csv \
    --in_dir ./retarget_in \
    --out_dir ./retarget_out \
    --has_header \
    --dry

Options:
  --dry          : Show what would be done (no files written)
  --overwrite    : Overwrite existing outputs (default: add numeric suffix)
  --has_header   : Interpret first row of each source CSV as a header
  --encoding ENC : File encoding (default: utf-8)
"""

import os
import csv
import argparse
from typing import List, Optional

def ensure_csv_ext(name: str) -> str:
    return name if name.lower().endswith(".csv") else f"{name}.csv"

def unique_path(path: str) -> str:
    """If path exists, append _1, _2, ... until unique."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    k = 1
    while True:
        cand = f"{base}_{k}{ext}"
        if not os.path.exists(cand):
            return cand
        k += 1

def clip_csv(
    in_path: str,
    out_path: str,
    start: int,
    end: Optional[int],
    has_header: bool = False,
    encoding: str = "utf-8",
) -> None:
    """
    Clip rows [start..end] (1-indexed, inclusive) from CSV at in_path and save to out_path.
    If has_header=True, preserves the first line as header and clips only the data rows.
    """
    # Read all lines first (preserve as text to avoid dtype surprises)
    with open(in_path, "r", encoding=encoding, errors="ignore", newline="") as f:
        reader = list(csv.reader(f))

    if not reader:
        raise RuntimeError(f"[{os.path.basename(in_path)}] Empty CSV.")

    header = None
    data = reader
    if has_header:
        header = reader[0]
        data = reader[1:]

    total = len(data)
    if total <= 0:
        # header-only file or truly empty
        clipped = []
    else:
        if end is None:
            end = total
        # 1-indexed inclusive -> python [start-1 : end]
        s_idx = max(0, int(start) - 1)
        e_idx = min(total, int(end))
        if s_idx >= e_idx or s_idx < 0 or e_idx > total:
            raise ValueError(f"[{os.path.basename(in_path)}] Invalid range: start={start}, end={end}, total={total}")
        clipped = data[s_idx:e_idx]

    # Write
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding=encoding, newline="") as f:
        writer = csv.writer(f)
        if has_header and header is not None:
            writer.writerow(header)
        writer.writerows(clipped)

def main():
    parser = argparse.ArgumentParser(description="Batch CSV clipper from segments CSV.")
    parser.add_argument("--csv", required=True, help="Segments CSV with columns: file,start,end[,out]")
    parser.add_argument("--in_dir", default=".", help="Input directory containing source CSVs")
    parser.add_argument("--out_dir", default="./out_csv", help="Output directory for clipped CSVs")
    parser.add_argument("--dry", action="store_true", help="Dry run (no files written)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite outputs if they exist")
    parser.add_argument("--has_header", action="store_true", help="Treat the first row of each source CSV as header")
    parser.add_argument("--encoding", default="utf-8", help="File encoding (default: utf-8)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # For auto seg numbering when 'out' is not specified
    seg_counts: dict[str, int] = {}

    # Read segments spec
    with open(args.csv, "r", encoding=args.encoding) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError("Segments CSV seems empty or malformed (no header).")
        headers = {h.strip().lower() for h in reader.fieldnames}
        required = {"file", "start", "end"}
        if not required.issubset(headers):
            raise RuntimeError(f"Segments CSV header must include at least: {sorted(required)}. Got: {sorted(headers)}")

        for row in reader:
            if not row or not (row.get("file") and row.get("start") and row.get("end")):
                continue

            file_name = row["file"].strip()
            try:
                start = int(row["start"])
                end = int(row["end"])
            except Exception:
                print(f"❌ Skip (bad start/end): {row}")
                continue

            custom_out = (row.get("out") or "").strip()

            in_path = os.path.join(args.in_dir, file_name)
            if not os.path.isfile(in_path):
                print(f"❌ Input not found: {in_path}")
                continue

            base = os.path.splitext(os.path.basename(file_name))[0]

            # Decide output relative path
            if custom_out:
                out_rel = ensure_csv_ext(custom_out)
            else:
                seg_counts.setdefault(base, 0)
                seg_counts[base] += 1
                out_rel = f"{base}_seg{seg_counts[base]}.csv"

            # Allow subfolders in `out` under out_dir (e.g., "walk/run1_seg1.csv")
            out_path = os.path.join(args.out_dir, out_rel)

            # Conflict handling
            if not args.overwrite and os.path.exists(out_path):
                out_path = unique_path(out_path)

            if args.dry:
                print(f"[DRY] {in_path}  ->  {out_path}   (rows {start}..{end}, header={args.has_header})")
                continue

            try:
                clip_csv(
                    in_path=in_path,
                    out_path=out_path,
                    start=start,
                    end=end,
                    has_header=args.has_header,
                    encoding=args.encoding,
                )
                print(f"✅ Saved: {out_path}   (rows {start}..{end}, header={args.has_header})")
            except Exception as e:
                print(f"❌ Error: {in_path} ({start}..{end}): {e}")

if __name__ == "__main__":
    main()
