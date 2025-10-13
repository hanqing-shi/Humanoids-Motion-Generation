"""
Batch BVH Clipper
- Read clip specs from a CSV (columns: file,start,end[,out])
- Clip frames [start..end] (1-indexed, inclusive) from each BVH
- Write result BVHs to out_dir with either `out` or auto `*_seg{k}.bvh` naming
- Preserves HIERARCHY; updates MOTION Frames: N; Frame Time: unchanged

Usage:
  python clip_bvh_batch.py \
    --csv segments.csv \
    --in_dir ../robot_data/ubisoft-laforge-animation-dataset/output/BVH \
    --out_dir ../../Humanoids-Motion-Generation/Lafan1_clipped \
    --dry

Options:
  --dry         : show what would be done, no files written
  --overwrite   : allow overwriting existing files (default: add numeric suffix)
"""

import os
import csv
import argparse
from typing import List

# ------------------------- Core BVH clipping -------------------------

def clip_bvh(in_path: str, out_path: str, start: int = 1, end: int | None = None) -> None:
    """
    Clip BVH frames [start..end] (1-indexed, end inclusive) and save to out_path.
    """
    with open(in_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.read().lstrip("\ufeff").splitlines()

    # Find MOTION section
    try:
        motion_idx = next(i for i, ln in enumerate(lines) if ln.strip().upper() == "MOTION")
    except StopIteration:
        raise RuntimeError(f"[{os.path.basename(in_path)}] MOTION section not found.")

    if motion_idx + 2 >= len(lines):
        raise RuntimeError(f"[{os.path.basename(in_path)}] Incomplete MOTION header.")

    frames_line = lines[motion_idx + 1].strip()
    time_line   = lines[motion_idx + 2].strip()

    if not frames_line.lower().startswith("frames:"):
        raise RuntimeError(f"[{os.path.basename(in_path)}] 'Frames:' line not found.")
    if not time_line.lower().startswith("frame time:"):
        raise RuntimeError(f"[{os.path.basename(in_path)}] 'Frame Time:' line not found.")

    frame_lines: List[str] = lines[motion_idx + 3:]
    total = len(frame_lines)

    if total <= 0:
        raise RuntimeError(f"[{os.path.basename(in_path)}] No frame data lines found.")

    # 1-indexed inclusive -> Python slice [start-1 : end]
    if end is None:
        end = total

    s_idx = max(0, int(start) - 1)
    e_idx = min(total, int(end))  # slice end is exclusive

    if s_idx >= e_idx or s_idx < 0 or e_idx > total:
        raise ValueError(f"[{os.path.basename(in_path)}] Invalid range: start={start}, end={end}, total={total}")

    clipped = frame_lines[s_idx:e_idx]
    newN = len(clipped)

    out_lines: List[str] = []
    out_lines.extend(lines[:motion_idx])   # keep HIERARCHY as-is
    out_lines.append("MOTION")
    out_lines.append(f"Frames: {newN}")
    out_lines.append(time_line)            # preserve frame time
    out_lines.extend(clipped)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(out_lines) + "\n")

# ------------------------- Utilities -------------------------

def ensure_bvh_ext(name: str) -> str:
    return name if name.lower().endswith(".bvh") else f"{name}.bvh"

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

# ------------------------- Main (batch) -------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch BVH clipper from CSV.")
    parser.add_argument("--csv", default="segments.csv", help="Path to CSV with columns: file,start,end[,out]")
    parser.add_argument("--in_dir", default=".", help="Input directory containing source .bvh files")
    parser.add_argument("--out_dir", default="./out_bvh", help="Output directory for clipped .bvh files")
    parser.add_argument("--dry", action="store_true", help="Dry run (no files written)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite if output file exists (default: add suffix)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # For auto seg numbering when 'out' is not specified
    seg_counts: dict[str, int] = {}

    # Read CSV
    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError("CSV seems empty or malformed (no header).")
        headers = {h.strip().lower() for h in reader.fieldnames}
        required = {"file", "start", "end"}
        if not required.issubset(headers):
            raise RuntimeError(f"CSV header must include at least: {sorted(required)}. Got: {sorted(headers)}")

        for row in reader:
            if not row or not row.get("file"):
                continue

            file_name = row["file"].strip()
            try:
                start = int(row["start"])
                end   = int(row["end"])
            except Exception:
                print(f"❌ Skip (bad start/end): {row}")
                continue

            custom_out = (row.get("out") or "").strip()

            in_path = os.path.join(args.in_dir, file_name)
            if not os.path.isfile(in_path):
                print(f"❌ Input not found: {in_path}")
                continue

            base = os.path.splitext(os.path.basename(file_name))[0]

            if custom_out:
                out_rel = ensure_bvh_ext(custom_out)
            else:
                seg_counts.setdefault(base, 0)
                seg_counts[base] += 1
                out_rel = f"{base}_seg{seg_counts[base]}.bvh"

            # Allow subfolders in `out` under out_dir (e.g., "walk/run1_seg1.bvh")
            out_path = os.path.join(args.out_dir, out_rel)

            # Conflict handling
            if not args.overwrite and os.path.exists(out_path):
                out_path = unique_path(out_path)

            if args.dry:
                print(f"[DRY] {in_path}  ->  {out_path}   (frames {start}..{end})")
                continue

            try:
                clip_bvh(in_path, out_path, start=start, end=end)
                print(f"✅ Saved: {out_path}   (frames {start}..{end})")
            except Exception as e:
                print(f"❌ Error: {in_path} ({start}..{end}): {e}")

if __name__ == "__main__":
    main()