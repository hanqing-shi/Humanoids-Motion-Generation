import os, glob
import numpy as np
import pandas as pd
import pinocchio.pinocchio_pywrap_default as pin
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import json

# ----------------------------- helpers -------------------------------- #

def _read_numeric_matrix(csv_path: str) -> np.ndarray:
    """Return a 2D numeric matrix from the input CSV, robust to headers."""
    df = pd.read_csv(csv_path)
    # take only numeric columns; guarantees 2D
    X = df.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return X

def q_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    """(T,4) wxyz -> (T,4) xyzw"""
    return np.column_stack([q_wxyz[:,1], q_wxyz[:,2], q_wxyz[:,3], q_wxyz[:,0]])

def q_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    """(T,4) xyzw -> (T,4) wxyz"""
    return np.column_stack([q_xyzw[:,3], q_xyzw[:,0], q_xyzw[:,1], q_xyzw[:,2]])

def quat_sign_continuous(q_wxyz: np.ndarray) -> np.ndarray:
    """Ensure quaternion sign continuity across time (no jumps by -q)."""
    q = q_wxyz.copy()
    # vectorized cumulative sign flip
    dots = np.sum(q[1:] * q[:-1], axis=1)
    flips = np.where(dots < 0.0, -1.0, 1.0)
    flips = np.concatenate([[1.0], flips]).astype(q.dtype)
    flips = np.cumprod(flips)[:, None]
    q *= flips
    # renormalize (cheap & safe)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q

def get_body_names_from_urdf(urdf_path: str):
    model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
    body_names = [f.name for f in model.frames if f.type == pin.FrameType.BODY]
    return body_names

def central_diff_first_last(X: np.ndarray, dt: float) -> np.ndarray:
    """Central difference with one-sided ends. X: (T, D)"""
    T = X.shape[0]
    V = np.zeros_like(X)
    if T >= 3:
        V[1:-1] = (X[2:] - X[:-2]) / (2.0 * dt)
        V[0]    = (X[1] - X[0]) / dt
        V[-1]   = (X[-1] - X[-2]) / dt
    elif T == 2:
        V[0] = V[1] = (X[1] - X[0]) / dt
    return V

def tri_moving_average(X: np.ndarray) -> np.ndarray:
    """3-point triangular moving average with 2-point at ends. X: (T, D)"""
    T = X.shape[0]
    if T == 0: return X
    Y = X.copy()
    if T >= 3:
        Y[1:-1] = (X[:-2] + X[1:-1] + X[2:]) / 3.0
        Y[0]    = (X[0] + X[1]) / 2.0
        Y[-1]   = (X[-1] + X[-2]) / 2.0
    elif T == 2:
        Y[0] = (X[0] + X[1]) / 2.0
        Y[1] = Y[0]
    return Y

def rot_mats_from_quat_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    """(T,4) wxyz -> (T,3,3) rotation matrices."""
    return R.from_quat(q_wxyz_to_xyzw(q_wxyz)).as_matrix()

import json

def write_json_output(state_array: np.ndarray, body_names: list[str], out_json_path: str):
    """
    Convert the flat state array into a structured dict and write JSON.
    Each body gets its own entry for pos/q/v/omega.
    """
    T = state_array.shape[0]
    nbodies = len(body_names)

    # Start building JSON structure
    json_data = {
        "metadata": {"frames": T, "num_bodies": nbodies},
        "root": {
            "position": state_array[:, 0:3].tolist(),
            "orientation_wxyz": state_array[:, 3:7].tolist(),
            "v_local": state_array[:, 7:10].tolist(),
            "omega_local": state_array[:, 10:13].tolist()
        },
        "bodies": {}
    }

    # offsets depend on your concatenation order in 'state'
    offset = 13
    stride = 3*nbodies + 4*nbodies + 3*nbodies + 3*nbodies
    body_pos_local = state_array[:, offset : offset + 3*nbodies]
    body_q_local   = state_array[:, offset + 3*nbodies : offset + 7*nbodies]
    body_v_local   = state_array[:, offset + 7*nbodies : offset + 10*nbodies]
    body_omega_loc = state_array[:, offset + 10*nbodies : offset + 13*nbodies]

    for i, name in enumerate(body_names):
        json_data["bodies"][name] = {
            "pos_local":  body_pos_local[:, 3*i:3*(i+1)].tolist(),
            "q_local_xyzw": body_q_local[:, 4*i:4*(i+1)].tolist(),
            "v_local":  body_v_local[:, 3*i:3*(i+1)].tolist(),
            "omega_local": body_omega_loc[:, 3*i:3*(i+1)].tolist(),
        }

    with open(out_json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"💾 saved JSON → {out_json_path}")

def world_to_local_batch(R_world_to_local: np.ndarray, X_world: np.ndarray) -> np.ndarray:
    """Apply per-frame rotation to (T,3N) blocks; returns (T,3N)."""
    T, D = X_world.shape
    N = D // 3
    Xl = np.empty_like(X_world)
    for t in range(T):
        Rt = R_world_to_local[t]               # (3,3)
        Wt = X_world[t].reshape(N, 3)
        Lt = (Rt @ Wt.T).T
        Xl[t] = Lt.reshape(-1)
    return Xl

# ----------------------- velocities (same method) ---------------------- #

def linear_velocities_from_pos_quat(positions: np.ndarray, q_wxyz: np.ndarray, dt: float):
    """
    Your original method: central diff on positions, then 3-pt moving average,
    then rotate to local with R^T.
    Returns: (v_world_smoothed, v_local, v_world_raw)
    """
    v_world = central_diff_first_last(positions, dt)
    v_world_ma = tri_moving_average(v_world)

    R_all = rot_mats_from_quat_wxyz(q_wxyz)     # (T,3,3)
    Rt_all = np.transpose(R_all, (0, 2, 1))     # (T,3,3)
    v_local = np.einsum('tij,tj->ti', Rt_all, v_world_ma)
    return v_world_ma, v_local, v_world

def angular_velocity_from_quat(q_wxyz: np.ndarray, dt: float):
    """
    Your original method: finite-difference on rotations (as rotvec),
    then 3-pt moving average; returns (omega_world_smoothed, omega_local).
    """
    q = quat_sign_continuous(q_wxyz)
    Rot = R.from_quat(q_wxyz_to_xyzw(q))

    T = q.shape[0]
    omega_world = np.zeros((T, 3))
    if T >= 3:
        fwd  = (Rot[2:]   * Rot[1:-1].inv()).as_rotvec() / dt
        back = (Rot[1:-1] * Rot[0:-2].inv()).as_rotvec() / dt
        omega_world[1:-1] = 0.5 * (fwd + back)
        omega_world[0]    = (Rot[1]  * Rot[0].inv()).as_rotvec()  / dt
        omega_world[-1]   = (Rot[-1] * Rot[-2].inv()).as_rotvec() / dt
    elif T == 2:
        omega_world[:] = (Rot[1] * Rot[0].inv()).as_rotvec() / dt

    omega_world_ma = tri_moving_average(omega_world)

    R_all = Rot.as_matrix()
    omega_local = np.einsum('tij,tj->ti', np.transpose(R_all, (0,2,1)), omega_world_ma)
    return omega_world_ma, omega_local

# ----------------------------- main logic ------------------------------ #

def process_csv(csv_path: str, urdf_path: str, dt: float, out_dir: str, show_plots: bool = False):
    df = pd.read_csv(csv_path, header=None).values
    T = df.shape[0]

    # 1) root position/orientation
    positions = df[:, 0:3].astype(np.float32, copy=False)
    q_wxyz    = df[:, 3:7].astype(np.float32, copy=False)
    joints = df[:,7:].astype(np.float32, copy=False)
    joint_count = joints.shape[1]

    # 2) root linear & angular velocities
    v_world_filt, v_local, v_world_raw = linear_velocities_from_pos_quat(positions, q_wxyz, dt)
    omega_world, omega_local = angular_velocity_from_quat(q_wxyz, dt)

    # 3) quick diagnostic plots (unchanged behavior)
    if show_plots:
        time = np.arange(T) * dt
        v_mag_raw  = np.linalg.norm(v_world_raw,  axis=1)
        v_mag_filt = np.linalg.norm(v_world_filt, axis=1)
        v_mag_sg   = savgol_filter(v_mag_raw, window_length=11, polyorder=3, mode='interp')

        fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        s0, s1 = 150, min(1000, T)

        axes[0].plot(time[s0:s1], v_mag_raw[s0:s1],  label="Unfiltered Global Velocity", alpha=.6)
        axes[1].plot(time[s0:s1], v_mag_filt[s0:s1], label="Filtered Global Velocity",   alpha=.6)
        axes[2].plot(time[s0:s1], v_mag_sg[s0:s1],   label="SG Filtered Global Velocity", alpha=.6)
        for ax in axes:
            ax.set_xlabel("Time")
            ax.set_ylabel("Velocity mag")
            ax.legend(); ax.grid(True)
        plt.tight_layout(); plt.show()

    # 4) Pinocchio FK (root is free-flyer)
    model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
    data  = model.createData()
    body_frame_ids = [fid for fid, f in enumerate(model.frames) if f.type == pin.FrameType.BODY]
    root_body_fid  = next((fid for fid in body_frame_ids if model.frames[fid].parentJoint == 1),
                          body_frame_ids[0])

    # prealloc
    Nframes = len(body_frame_ids)
    body_pos_world = np.empty((T, 3 * Nframes), dtype=np.float64)
    body_pos_local = np.empty_like(body_pos_world)
    body_q_world   = np.empty((T, 4 * Nframes), dtype=np.float64)  # xyzw packed
    body_q_local   = np.empty_like(body_q_world)

    # loop frames once; pack as we go
    for t in range(T):
        q_full = np.zeros(model.nq, dtype=np.float64)
        q_full[:3]  = df[t, 0:3]        # xyz
        q_full[3:7] = df[t, 3:7]        # wxyz
        q_full[7:]  = df[t, 7:]         # joints

        pin.forwardKinematics(model, data, q_full)
        pin.updateFramePlacements(model, data)

        M_root_inv = data.oMf[root_body_fid].inverse()

        # collect globals + locals
        Pw = np.empty((Nframes, 3)); Pl = np.empty_like(Pw)
        Qw = np.empty((Nframes, 4)); Ql = np.empty_like(Qw)  # xyzw
        for k, fid in enumerate(body_frame_ids):
            M_world = data.oMf[fid]
            Pw[k] = M_world.translation
            M_local = M_root_inv * M_world
            Pl[k] = M_local.translation
            Qw[k] = R.from_matrix(M_world.rotation).as_quat()
            Ql[k] = R.from_matrix(M_local.rotation).as_quat()

        body_pos_world[t] = Pw.reshape(-1)
        body_pos_local[t] = Pl.reshape(-1)
        body_q_world[t]   = Qw.reshape(-1)
        body_q_local[t]   = Ql.reshape(-1)

    # 5) body linear velocities (world → local with root R^T)
    body_v_world = central_diff_first_last(body_pos_world, dt)
    R_all = rot_mats_from_quat_wxyz(q_wxyz)
    body_v_local = world_to_local_batch(np.transpose(R_all, (0, 2, 1)), body_v_world)

    # 6) body angular velocities (LOCAL) per body (fixes previous indentation bug)
    Nbodies = body_q_local.shape[1] // 4
    body_omega_local = np.zeros((T, Nbodies * 3), dtype=np.float64)
    for j in range(Nbodies):
        q_local_xyzw_j = body_q_local[:, 4*j:4*(j+1)]
        q_local_wxyz_j = q_xyzw_to_wxyz(q_local_xyzw_j)
        _, omega_local_j = angular_velocity_from_quat(q_local_wxyz_j, dt)
        body_omega_local[:, 3*j:3*(j+1)] = omega_local_j

    # 7) final state (unchanged layout)
    state = np.hstack((
        positions,              # 3
        q_wxyz,                 # 4
        v_local,                # 3
        omega_local,            # 3
        body_pos_local,         # 3*N
        body_q_local,           # 4*N (xyzw)
        body_v_local,           # 3*N
        body_omega_local,       # 3*N
        joints
    ))
 
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(csv_path))[0]
    T_here = state.shape[0]  # same as T

    model_state = np.hstack([
        positions,            # root_x,root_y,root_z
        q_wxyz,               # root_w,root_qx,root_qy,root_qz
        v_local,              # root_vx,root_vy,root_vz
        omega_local,          # root_omx,root_omy,root_omz
        joints                # joint_1..joint_N
    ])
    joint_count = joints.shape[1]
    cols = (["root_x","root_y","root_z",
             "root_w","root_qx","root_qy","root_qz",
             "root_vx","root_vy","root_vz",
             "root_omx","root_omy","root_omz"] +
            [f"joint_{i+1}" for i in range(joint_count)])
    df_model = pd.DataFrame(model_state, columns=cols)
    df_model.insert(0, "frame", np.arange(T_here, dtype=int))
    out_model_csv = os.path.join(out_dir, f"{base}_modelstate.csv")
    df_model.to_csv(out_model_csv, index=False)
    print(f"✅ saved model-ready → {out_model_csv}  (T={T_here}, joints={joint_count})")

    # --- original outputs (training csv + json) ---
    body_names = get_body_names_from_urdf(urdf_path)
    out_path = os.path.join(out_dir, f"{base}_training.csv")
    out_path_json = os.path.join(out_dir, f"{base}_training.json")
    np.savetxt(out_path, state, delimiter=",")
    write_json_output(state, body_names, out_path_json)
    print(f"✅ saved {base} → {out_path}")

    os.makedirs(out_dir, exist_ok=True)
    body_names = get_body_names_from_urdf(urdf_path)
    base = os.path.splitext(os.path.basename(csv_path))[0]
    out_path = os.path.join(out_dir, f"{base}_training.csv")
    out_path_json = os.path.join(out_dir, f"{base}_training.json")
    np.savetxt(out_path, state, delimiter=",")
    write_json_output(state, body_names, out_path_json)
    print(f"✅ saved {base} → {out_path}")

def batch_process(in_dir: str, urdf: str, out_dir: str, dt: float = 1/30):
    csvs = sorted(glob.glob(os.path.join(in_dir, "*.csv")))
    if not csvs:
        print(f"(no CSVs found in {in_dir})"); return
    for csv_path in csvs:
        try:
            process_csv(csv_path, urdf, dt, out_dir)
        except Exception as e:
            print(f"❌ FAILED: {csv_path}\n   → {e}")

if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser(description="Build flattened training features (LOCAL velocities) for all CSV clips in a folder.")
    ap.add_argument("--in_dir",  default=os.path.join("test_data","CSV_INPUT"), help="Folder with config CSVs from bvh_to_csv.py")
    ap.add_argument("--urdf",    default=os.path.join("test_data","g1","g1_29dof_rev_1_0.urdf"), help="URDF path (free-flyer compatible)")
    ap.add_argument("--out_dir", default=os.path.join("test_data","Training_Output"), help="Where to write *_features.csv")
    ap.add_argument("--dt", type=float, default=1/30, help="Frame time in seconds (e.g., 1/30)")
    args = ap.parse_args()
    batch_process(args.in_dir, args.urdf, args.out_dir, args.dt)
