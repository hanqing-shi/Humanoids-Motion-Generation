import os
import numpy as np
import pandas as pd
from bvh import Bvh
from scipy.spatial.transform import Rotation as R
import math

def Rx(a): c,s=math.cos(a),math.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def Ry(a): c,s=math.cos(a),math.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def Rz(a): c,s=math.cos(a),math.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def R_from_row(row_deg, order):
    ang = dict(zip(order, np.deg2rad(row_deg)))
    Rm = np.eye(3)
    for name in order:
        Rm = Rm @ (Rx(ang[name]) if name=='Xrotation'
                   else Ry(ang[name]) if name=='Yrotation'
                   else Rz(ang[name]))
    return Rm


def process_bvh_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        bvh = Bvh(f.read().lstrip("\ufeff"))

    root = bvh.get_joints_names()[0]
    ch = bvh.joint_channels(root)  # ['Xposition','Yposition','Zposition','Zrotation','Yrotation','Xrotation']
    T = bvh.nframes
    dt = bvh.frame_time

    M = np.array(bvh.frames_joint_channels(root, ch), dtype=float)
    positions = M[:, :3]
    euler_deg = M[:, 3:]
    rot_order = ch[3:]

    # Rotation matrices
    R_all = np.stack([R_from_row(euler_deg[t], rot_order) for t in range(T)], axis=0)
    Rot = R.from_matrix(R_all)

    # Angular velocity (world)
    omega_world = np.zeros((T, 3))
    if T > 2:
        delta_fwd = (Rot[2:] * Rot[1:-1].inv()).as_rotvec() / dt
        delta_back = (Rot[1:-1] * Rot[:-2].inv()).as_rotvec() / dt
        omega_world[1:-1] = 0.5 * (delta_fwd + delta_back)
        omega_world[0] = ((Rot[1] * Rot[0].inv()).as_rotvec()) / dt
        omega_world[-1] = ((Rot[-1] * Rot[-2].inv()).as_rotvec()) / dt

    # Map to local frame
    R_T = np.transpose(R_all, (0, 2, 1))
    omega_local = np.einsum('tij,tj->ti', R_T, omega_world)

    # Linear global velocity
    v_world = np.zeros_like(positions)
    if T > 2:
        v_world[1:-1] = (positions[2:] - positions[:-2]) / (2 * dt)
        v_world[0] = (positions[1] - positions[0]) / dt
        v_world[-1] = (positions[-1] - positions[-2]) / dt

    # To local frame
    v_local = np.einsum('tij,tj->ti', R_T, v_world)

    # Twist about local Y
    twist_vec_local = omega_local[:, 1].reshape(-1, 1)

    # Quaternions (wxyz)
    quat_xyzw = R.from_matrix(R_all).as_quat()
    quat_wxyz = np.column_stack([
        quat_xyzw[:, 3], quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2]
    ])

    q = np.hstack((positions, quat_wxyz))
    qdot = np.hstack((v_local, omega_local))
    x = np.hstack((q, qdot, twist_vec_local))

    columns = [
        "pos_x", "pos_y", "pos_z",
        "quat_w", "quat_x", "quat_y", "quat_z",
        "vel_x", "vel_y", "vel_z",
        "omega_x", "omega_y", "omega_z",
        "twist_y"
    ]

    time = np.arange(T) * dt
    df = pd.DataFrame(x, columns=columns)
    df.insert(0, "time", time)

    return df


def process_folder(folder=".",out_folder='.'):
    for file in os.listdir(folder):
        if file.lower().endswith(".bvh"):
            in_path = os.path.join(folder, file)
            out_path = os.path.join(out_folder, file)
            out_path = os.path.splitext(out_path)[0] + ".csv"
            try:
                df = process_bvh_file(in_path)
                df.to_csv(out_path, index=False)
                print(f"✅ Saved {out_path}")
            except Exception as e:
                print(f"❌ Error processing {file}: {e}")


if __name__ == "__main__":
    process_folder("./Lafan1_clipped","./Lafan1_clipped_label")
