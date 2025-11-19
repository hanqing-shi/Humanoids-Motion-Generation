import os, glob
import numpy as np
import pandas as pd
import pinocchio as pin
from pathlib import Path

def row_to_q(row: np.ndarray) -> np.ndarray:
    """CSV row [xyz, wxyz, joints...] -> Pinocchio q (free-flyer + joints)."""
    xyz, wxyz = row[:3], row[3:7]
    wxyz = wxyz / np.linalg.norm(wxyz)  # safety
    return np.r_[xyz, wxyz, row[7:]]

def get_body_frame_ids(model: pin.Model) -> list[int]:
    """Return all BODY frame ids (exclude visuals/sensors)."""
    return [fid for fid, f in enumerate(model.frames) if f.type == pin.FrameType.BODY]

def find_root_body_frame_id(model: pin.Model, body_fids: list[int]) -> int:
    """A BODY frame attached to the free-flyer joint (id=1)."""
    for fid in body_fids:
        if model.frames[fid].parent == 1:
            return fid
    return body_fids[0]

def headers_for_clip(model: pin.Model, body_fids: list[int], J: int) -> list[str]:
    """Column headers in the order we concatenate below."""
    cols = []
    # 1 Root positions
    cols += ["root_x","root_y","root_z"]
    # 2 Root orientations (quat wxyz)
    cols += ["root_qw","root_qx","root_qy","root_qz"]
    # 3 Root linear vel (LOCAL)
    cols += ["root_vx_local","root_vy_local","root_vz_local"]
    # 4 Root angular vel (LOCAL)
    cols += ["root_wx_local","root_wy_local","root_wz_local"]

    def add_body_cols(prefix, dim):
        for fid in body_fids:
            name = model.frames[fid].name
            for d in range(dim):
                cols.append(f"{name}_{prefix}{d}")

    # 5 Body positions (world) -> 3*N
    add_body_cols("pos_", 3)
    # 6 Body orientations (quat xyzw, world) -> 4*N
    for fid in body_fids:
        name = model.frames[fid].name
        cols += [f"{name}_quat_x", f"{name}_quat_y", f"{name}_quat_z", f"{name}_quat_w"]
    # 7 Body linear vel (LOCAL) -> 3*N
    add_body_cols("v_local_", 3)
    # 8 Body angular vel (LOCAL) -> 3*N
    add_body_cols("w_local_", 3)
    # 9 Joint angles (URDF order, radians)
    for j in range(J):
        cols.append(f"joint_{j}_pos")
    # 10 Joint velocities
    for j in range(J):
        cols.append(f"joint_{j}_vel")

    return cols

def process_csv(csv_path: str, urdf_path: str, dt: float, out_dir: str):
    model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
    data  = model.createData()

    Qraw = np.loadtxt(csv_path, delimiter=',')         # (T, 7+J)
    if Qraw.ndim == 1:
        Qraw = Qraw[None, :]
    T = Qraw.shape[0]
    J = Qraw.shape[1] - 7

    Q = np.vstack([row_to_q(r) for r in Qraw])         # (T, nq)

    V = np.zeros((T, model.nv))
    for t in range(T-1):
        V[t] = pin.difference(model, Q[t], Q[t+1]) / dt
    V[-1] = V[-2] if T > 1 else V[-1]

    body_fids = get_body_frame_ids(model)
    root_body_fid = find_root_body_frame_id(model, body_fids)
    N = len(body_fids)

    root_positions            = np.zeros((T, 3))
    root_orient_wxyz          = np.zeros((T, 4))
    root_linear_vel_local     = np.zeros((T, 3))
    root_angular_vel_local    = np.zeros((T, 3))

    body_positions_world      = np.zeros((T, N, 3))
    body_orient_xyzw_world    = np.zeros((T, N, 4))
    body_linear_vel_local     = np.zeros((T, N, 3))
    body_angular_vel_local    = np.zeros((T, N, 3))

    joint_pos = Qraw[:, 7:]               # (T, J)
    joint_vel = V[:, 6:]                  # skip free-flyer nv=6

    for t in range(T):
        q, v = Q[t], V[t]
        pin.forwardKinematics(model, data, q, v)
        pin.updateFramePlacements(model, data)
        pin.framesForwardKinematics(model, data, q, v)

        # 1–2: Root pose (global/world) straight from CSV
        root_positions[t]   = Qraw[t, :3]
        root_orient_wxyz[t] = Qraw[t, 3:7]

        # 3–4: Root velocities in LOCAL frame of the root body
        v_root = pin.getFrameVelocity(model, data, root_body_fid, pin.ReferenceFrame.LOCAL)
        root_linear_vel_local[t]  = v_root.linear
        root_angular_vel_local[t] = v_root.angular

        # 5–8: Body poses (WORLD) & velocities (LOCAL)
        for k, fid in enumerate(body_fids):
            M = data.oMf[fid]                                    # world pose
            body_positions_world[t, k] = M.translation
            # quaternion (Pin: coeffs() -> xyzw)
            body_orient_xyzw_world[t, k] = pin.Quaternion(M.rotation).coeffs()

            v_local = pin.getFrameVelocity(model, data, fid, pin.ReferenceFrame.LOCAL)
            body_linear_vel_local[t, k]  = v_local.linear
            body_angular_vel_local[t, k] = v_local.angular

    # Flatten groups 5–8 and concatenate all groups horizontally
    features = np.hstack([
        root_positions,                      # 3
        root_orient_wxyz,                    # 4
        root_linear_vel_local,               # 3
        root_angular_vel_local,              # 3
        body_positions_world.reshape(T, -1),     # 3N
        body_orient_xyzw_world.reshape(T, -1),   # 4N
        body_linear_vel_local.reshape(T, -1),    # 3N
        body_angular_vel_local.reshape(T, -1),   # 3N
        joint_pos,                           # J
        joint_vel                            # J
    ])

    # Column headers
    headers = headers_for_clip(model, body_fids, J)

    # Save per-clip CSV
    clip = os.path.splitext(os.path.basename(csv_path))[0]
    out_csv = os.path.join(out_dir, f"{clip}_features.csv")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Use pandas so headers are included
    df = pd.DataFrame(features, columns=headers)
    df.to_csv(out_csv, index=False)
    print(f"✅ {clip}: saved {features.shape} → {out_csv}")

def batch_process(in_dir: str, urdf: str, out_dir: str, dt: float = 1/30):
    csvs = sorted(glob.glob(os.path.join(in_dir, "*.csv")))
    if not csvs:
        print(f"(no CSVs found in {in_dir})")
        return
    for csv_path in csvs:
        process_csv(csv_path, urdf, dt, out_dir)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build flattened training features (LOCAL velocities) for all CSV clips in a folder.")
    ap.add_argument("--in_dir", required=True, help="Folder with config CSVs from bvh_to_csv.py")
    ap.add_argument("--urdf", required=True, help="URDF path (free-flyer compatible)")
    ap.add_argument("--out_dir", required=True, help="Where to write *_features.csv")
    ap.add_argument("--dt", type=float, default=1/30, help="Frame time in seconds (e.g., 1/30)")
    args = ap.parse_args()

    batch_process(args.in_dir, args.urdf, args.out_dir, args.dt)
