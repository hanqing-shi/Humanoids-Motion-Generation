import os, glob
import numpy as np
import pandas as pd
import pinocchio as pin
from pathlib import Path
from scipy.spatial.transform import Rotation as R

def linear_velocities_from_pos_quat(positions: np.ndarray, q_wxyz: np.ndarray, dt: float):
    T = positions.shape[0]
    v_world = np.zeros_like(positions)

    if T>=3:
        v_world[1:-1] = (positions[2:] - positions[:-2]) / (2.0 * dt) # central difference
        v_world[0] = (positions[1] - positions[0]) / dt
        v_world[-1] = (positions[-1] - positions[-2]) / dt
    elif T==2:
        v_world[0]  = (positions[1] - positions[0]) / dt
        v_world[1]  = v_world[0]
    else:
        pass
    q_xyzw = np.column_stack([q_wxyz[:,1], q_wxyz[:,2], q_wxyz[:,3], q_wxyz[:,0]]) # for scipy indexing
    Rot = R.from_quat(q_xyzw)
    R_all = Rot.as_matrix()
    v_local = np.einsum('tij,tj->ti', np.transpose(R_all, (0,2,1)), v_world)
    return v_world, v_local

def angular_velocity_from_quat(q_wxyz: np.ndarray, dt: float):
    T = q_wxyz.shape[0]
    omega_world = np.zeros((T, 3))
    q = q_wxyz.copy()
    for t in range(1, T):
        if np.dot(q[t], q[t-1]) < 0.0:
            q[t] = -q[t]
    q_xyzw = np.column_stack([q[:,1], q[:,2], q[:,3], q[:,0]]) # for scipy indexing
    Rot = R.from_quat(q_xyzw)

    if T >= 3:
        delta_fwd  = (Rot[2:]   * Rot[1:-1].inv()).as_rotvec() / dt
        delta_back = (Rot[1:-1] * Rot[0:-2].inv()).as_rotvec()  / dt
        omega_world[1:-1] = 0.5 * (delta_fwd + delta_back)
        omega_world[0]    = (Rot[1]  * Rot[0].inv()).as_rotvec()   / dt
        omega_world[-1]   = (Rot[-1] * Rot[-2].inv()).as_rotvec()  / dt
    elif T == 2:
        omega_world[0]  = (Rot[1] * Rot[0].inv()).as_rotvec() / dt
        omega_world[1]  = omega_world[0]
    R_all = Rot.as_matrix()
    omega_local = np.einsum('tij,tj->ti', np.transpose(R_all, (0,2,1)), omega_world)
    return omega_world, omega_local

def process_csv(csv_path: str, urdf_path: str, dt: float, out_dir: str):
    file = pd.read_csv(csv_path, header=None)
    df = file.values
    T = df.shape[0]
    '''position and orientation'''
    positions = df[:,0:3]
    q_wxyz = df[:, 3:7]
    '''velocities'''
    v_world, v_local = linear_velocities_from_pos_quat(positions, q_wxyz, dt)
    omega_world, omega_local = angular_velocity_from_quat(q_wxyz, dt)

    '''pinocchio forward kinematics'''
    model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
    data  = model.createData()
    body_frame_ids = [fid for fid,f in enumerate(model.frames) if f.type == pin.FrameType.BODY]
    root_body_fid = next((fid for fid in body_frame_ids if model.frames[fid].parentJoint == 1), body_frame_ids[0])

    '''body pos'''
    body_pos_local = []
    body_pos_world = []

    for t in range(T):
        q = np.zeros(model.nq)
        q[:3] = df[t,0:3] #xyz
        q[3:7] = df[t, 3:7] #wxyz
        q[7:] = df[t,7:] # joint angles

        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        M_root_inv = data.oMf[root_body_fid].inverse()

        body_positions_global = []
        body_positions_local = []

        for fid in body_frame_ids:
            M_world = data.oMf[fid]
            body_positions_global.append(M_world.translation)
            M_local = M_root_inv * M_world
            body_positions_local.append(M_local.translation)   

        body_pos_world.append(np.concatenate(body_positions_global))
        body_pos_local.append(np.concatenate(body_positions_local))

    body_pos_world = np.vstack(body_pos_world)
    body_pos_local = np.vstack(body_pos_local)

    '''body pos relative to root'''
    #body_pos_rel_root = body_pos_world - np.repeat(positions, body_pos_world.shape[1]//3, axis=1)

    '''body v world, local, and relative to root'''
    '''
    body_v_world = np.zeros_like(body_pos_world)
    body_v_local = np.zeros_like(body_pos_local)
    if T >= 3:
        body_v_world[1:-1] = (body_pos_world[2:] - body_pos_world[:-2]) / (2.0 * dt)
        body_v_world[0] = (body_pos_world[1] - body_pos_world[0]) / dt
        body_v_world[-1] = (body_pos_world[-1] - body_pos_world[-2]) / dt
    elif T == 2:
        body_v_world[0] = (body_pos_world[1] - body_pos_world[0]) / dt
        body_v_world[1] = body_v_world[0]
    N = body_pos_world.shape[1]//3
    pos_rep = np.repeat(positions, N, axis=1)
    vel_rep = np.repeat(v_world, N, axis=1)
    body_pos_rel_root = body_pos_world - pos_rep
    body_v_rel_root = body_v_world - vel_rep
    '''
    body_v_world = np.zeros_like(body_pos_world)
    if T>=3:
        body_v_world[1:-1] = (body_pos_world[2:] - body_pos_world[:-2]) / (2.0 * dt)
        body_v_world[0]    = (body_pos_world[1] - body_pos_world[0]) / dt
        body_v_world[-1]   = (body_pos_world[-1] - body_pos_world[-2]) / dt
    elif T == 2:
        body_v_world[0] = (body_pos_world[1] - body_pos_world[0]) / dt
        body_v_world[1] = body_v_world[0]
    
    q_xyzw = np.column_stack([q_wxyz[:,1], q_wxyz[:,2], q_wxyz[:,3], q_wxyz[:,0]])
    Rot = R.from_quat(q_xyzw).as_matrix()

    N = body_pos_world.shape[1] // 3

    body_pos_local = np.empty_like(body_pos_world)
    body_v_local   = np.empty_like(body_v_world)

    for t in range(T):
        Rt = Rot[t].T                      # world->root (local) rotation
        Pw = body_pos_world[t].reshape(N,3)
        Vw = body_v_world[t].reshape(N,3)
        Pl = (Rt @ Pw.T).T                 # (N,3)
        Vl = (Rt @ Vw.T).T                 # (N,3)
        body_pos_local[t] = Pl.reshape(-1)
        body_v_local[t]   = Vl.reshape(-1)

    

    #body_v_world, body_v_local = linear_velocities_from_pos_quat(body_pos_world, q_wxyz, dt)

    state = np.hstack((positions, q_wxyz, v_local, omega_local, body_pos_local, body_v_local, body_q, body_omega))
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(csv_path))[0]
    out_path = os.path.join(out_dir, f"{base}_training.csv")
    np.savetxt(out_path, state, delimiter=",")
    print(f"✅ saved {base} → {out_path}")

def batch_process(in_dir: str, urdf: str, out_dir: str, dt: float = 1/30):
    csvs = sorted(glob.glob(os.path.join(in_dir, "*.csv")))
    if not csvs:
        print(f"(no CSVs found in {in_dir})")
        return
    for csv_path in csvs:
        try:
            process_csv(csv_path, urdf, dt, out_dir)
        except Exception as e:
            print(f"❌ FAILED: {csv_path}\n   → {e}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build flattened training features (LOCAL velocities) for all CSV clips in a folder.")
    ap.add_argument("--in_dir", required=True, help="Folder with config CSVs from bvh_to_csv.py")
    ap.add_argument("--urdf", required=True, help="URDF path (free-flyer compatible)")
    ap.add_argument("--out_dir", required=True, help="Where to write *_features.csv")
    ap.add_argument("--dt", type=float, default=1/30, help="Frame time in seconds (e.g., 1/30)")
    args = ap.parse_args()

    batch_process(args.in_dir, args.urdf, args.out_dir, args.dt)
