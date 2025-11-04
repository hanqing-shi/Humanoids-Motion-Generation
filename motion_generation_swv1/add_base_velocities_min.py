#!/usr/bin/env python3
# Minimal base velocity appender: vx, vy, wz only.
import argparse, os, glob, numpy as np

def looks_like_header(first_line: str) -> bool:
    toks = [t.strip() for t in first_line.replace('\t', ',').split(',')]
    def is_num(s: str) -> bool:
        try: float(s); return True
        except Exception: return False
    return any((t != '' and not is_num(t)) for t in toks)

def load_csv(path: str, col_offset: int = 0):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        first = f.readline()
    skip = 1 if looks_like_header(first) else 0
    data = np.genfromtxt(path, delimiter=',', skip_header=skip)
    if data.ndim == 1: data = data.reshape(1, -1)
    header = None
    if skip == 1: header = [t.strip() for t in first.replace('\t', ',').split(',')]
    if col_offset > 0:
        if data.shape[1] <= col_offset: raise ValueError(f"col_offset={col_offset} >= ncols={data.shape[1]}")
        data = data[:, col_offset:]
        if header is not None: header = header[col_offset:]
    return data, header

def save_csv(path: str, arr: np.ndarray, header=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if header is None:
        np.savetxt(path, arr, delimiter=',', fmt='%.10g')
    else:
        with open(path, 'w', encoding='utf-8', newline='') as f:
            f.write(','.join(header) + '\n')
            np.savetxt(f, arr, delimiter=',', fmt='%.10g')

def median_dt_from_timecol(raw_path: str, time_idx: int) -> float:
    with open(raw_path, 'r', encoding='utf-8', errors='ignore') as f:
        first = f.readline(); skip = 1 if looks_like_header(first) else 0
    t = np.genfromtxt(raw_path, delimiter=',', skip_header=skip, usecols=(time_idx,))
    if t.ndim == 0: t = np.array([t])
    if t.size < 2: raise ValueError("Not enough rows to estimate dt.")
    diffs = np.diff(t); diffs = diffs[np.isfinite(diffs)]; diffs = diffs[diffs > 0]
    if diffs.size == 0: raise ValueError("Time column does not increase; cannot estimate dt.")
    return float(np.median(diffs))

def central_diff(x: np.ndarray, dt: float) -> np.ndarray:
    v = np.empty_like(x); n = len(x)
    if n == 0: return np.array([], dtype=x.dtype)
    if n == 1: return np.zeros_like(x)
    v[0] = (x[1]-x[0])/dt; v[-1] = (x[-1]-x[-2])/dt
    if n > 2: v[1:-1] = (x[2:]-x[:-2])/(2*dt)
    return v

def quat_to_yaw(qw, qx, qy, qz):
    return np.arctan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))

def compute_base_velocities(data: np.ndarray, dt: float, x_idx=0, y_idx=1, qw_idx=3, qx_idx=4, qy_idx=5, qz_idx=6):
    if data.shape[1] <= max(x_idx, y_idx, qw_idx, qx_idx, qy_idx, qz_idx): 
        raise ValueError("Not enough columns.")
    x, y = data[:, x_idx], data[:, y_idx]
    qw, qx, qy, qz = data[:, qw_idx], data[:, qx_idx], data[:, qy_idx], data[:, qz_idx]
    vx = central_diff(x, dt); vy = central_diff(y, dt)
    yaw_unwrapped = np.unwrap(quat_to_yaw(qw, qx, qy, qz)); wz = central_diff(yaw_unwrapped, dt)
    return vx, vy, wz

def process_one(path_in: str, out_dir: str, dt: float, col_offset: int, time_idx: int,
                x_idx: int, y_idx: int, qw_idx: int, qx_idx: int, qy_idx: int, qz_idx: int, suffix="_vel"):
    data, header = load_csv(path_in, col_offset=col_offset)
    if time_idx >= 0:
        est_dt = median_dt_from_timecol(path_in, time_idx=time_idx)
        print(f"[info] Estimated dt from time column {time_idx}: {est_dt:.6f} s"); dt = est_dt
    vx, vy, wz = compute_base_velocities(data, dt, x_idx, y_idx, qw_idx, qx_idx, qy_idx, qz_idx)
    out = np.concatenate([data, vx[:,None], vy[:,None], wz[:,None]], axis=1)
    if header is not None: header = header + ["vx","vy","wz"]

    base_name = os.path.splitext(os.path.basename(path_in))[0]
    out_path = os.path.join(out_dir, base_name + suffix + ".csv")
    save_csv(out_path, out, header=header)
    print(f"✅ Saved: {out_path}  (dt={dt:.6f})")

def main():
    ap = argparse.ArgumentParser(description="Append ONLY base velocities (vx, vy, wz) to CSV(s) and save to separate dir.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", type=str, help="Input CSV file")
    src.add_argument("--dir", type=str, help="Directory to process all *.csv")
    ap.add_argument("--out_dir", type=str, default="retarget_out_vel", help="Output directory")
    ap.add_argument("--dt", type=float, default=(1.0/30.0), help="Sampling period in seconds (default 1/30 ≈ LAFAN1)")
    ap.add_argument("--time_idx", type=int, default=-1, help="Absolute time column index (before col_offset) for dt estimation; -1 disables.")
    ap.add_argument("--col_offset", type=int, default=0, help="Skip N leading columns (e.g., time index)")
    ap.add_argument("--x_idx", type=int, default=0)
    ap.add_argument("--y_idx", type=int, default=1)
    ap.add_argument("--qw_idx", type=int, default=3)
    ap.add_argument("--qx_idx", type=int, default=4)
    ap.add_argument("--qy_idx", type=int, default=5)
    ap.add_argument("--qz_idx", type=int, default=6)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    paths = [args.file] if args.file else sorted(glob.glob(os.path.join(args.dir, "*.csv")))
    if not paths: raise SystemExit("No CSVs found.")

    for p in paths:
        try:
            process_one(p, args.out_dir, args.dt, args.col_offset, args.time_idx,
                        args.x_idx, args.y_idx, args.qw_idx, args.qx_idx, args.qy_idx, args.qz_idx)
        except Exception as e:
            print(f"❌ Error on {p}: {e}")

if __name__ == "__main__":
    main()
