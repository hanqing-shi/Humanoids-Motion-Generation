# robot_motion_viewer.py
# Visualize rollout CSVs (training export OR cleaned) with Pinocchio + Rerun.
# Usage examples:
#   python robot_motion_viewer.py --csv rollout_eval_epoch005.csv --urdf "<path>\robot_description\g1\g1_29dof_rev_1_0.urdf" --pkg "<path>\robot_description\g1" --fps 30
#   python robot_motion_viewer.py --csv visualize\rollout_eval_epoch005.csv --urdf "..." --pkg "..." --fps 30

import argparse
import os
import numpy as np
import pandas as pd

import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from scipy.spatial.transform import Rotation as R

import rerun as rr  # pip install rerun-sdk


def load_rollout(csv_path: str) -> np.ndarray:
    """
    Load a rollout CSV that may be:
      (A) Training export: has header + 'frame' column as first column.
      (B) Cleaned version: no header, no frame column.

    Returns:
      arr : (T, D) float array with columns laid out as in training export:
            [root_x, root_y, root_z,
             root_w, root_qx, root_qy, root_qz,
             root_vx, root_vy, root_vz,
             root_omx, root_omy, root_omz,
             joint_1, joint_2, ...]
    """
    # Try reading with header; if it fails (or all columns are numeric unnamed),
    # fallback to header=None and treat as cleaned.
    try:
        df = pd.read_csv(csv_path)
        # If the first column looks like 'frame', drop it:
        if "frame" in df.columns[0].lower():
            df = df.drop(columns=[df.columns[0]])
        # If columns are strings like 'root_x', great; otherwise they’ll be numeric names.
        arr = df.to_numpy(dtype=float)
    except Exception:
        # No header case:
        df = pd.read_csv(csv_path, header=None)
        arr = df.to_numpy(dtype=float)
    return arr


def to_pinocchio_q(row: np.ndarray) -> np.ndarray:
    """
    Convert one row from the training layout to Pinocchio free-flyer q:
      training layout:
        [0:3]=pos (x,y,z),
        [3:7]=quat (w,x,y,z),
        [7:13]=root v/ω (unused for q),
        [13:]=joints
      pinocchio q for free-flyer:
        [x,y,z, qx,qy,qz,qw, joints...]

    Returns:
      q : (7 + n_joints,)
    """
    x, y, z = row[0:3]
    # training export uses (w, x, y, z) at indices 3..6
    w, qx, qy, qz = row[3:7]
    joints = row[13:]
    # Pinocchio expects base quaternion as (x, y, z, w) *in the DoF order 3..6 = qx,qy,qz,qw
    q_free = np.array([x, y, z, qx, qy, qz, w], dtype=float)
    return np.concatenate([q_free, joints.astype(float)], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to rollout CSV (training export or cleaned).")
    ap.add_argument("--urdf", required=True, help="Path to robot URDF.")
    ap.add_argument("--pkg", required=True, help="Package dir for meshes/resources (Pinocchio search path).")
    ap.add_argument("--fps", type=float, default=30.0, help="Playback rate (frames per second).")
    ap.add_argument("--name", default="robot", help="Entity name in the viewer.")
    args = ap.parse_args()

    # Load CSV (handles both header+frame and clean variants)
    arr = load_rollout(args.csv)
    T, D = arr.shape
    if D < 14:
        raise ValueError(f"Unexpected CSV shape {arr.shape}. Need at least root pose(7) + root v/ω(6) + joints.")

    # Build robot
    # Important: Pinocchio BuildFromURDF expects package dirs as a LIST.
    robot = RobotWrapper.BuildFromURDF(args.urdf, [args.pkg], pin.JointModelFreeFlyer())
    model, data = robot.model, robot.data

    # Prepare Rerun
    rr.init("Robot Rollout Viewer", spawn=True)
    # Log a clear time sequence (frame count)
    rr.set_time_sequence("frame", 0)

    # We’ll draw:
    #   • The root trajectory in world (a line strip)
    #   • Each visual frame of the robot as a transform3D at each timestep
    #
    # Collect all frame names from Pinocchio's visual model:
    vis_frames = []
    for go in robot.visual_model.geometryObjects:
        # Rerun entity paths like: 'robot/links/<name>'
        name = go.name[:-2] if go.name.endswith("_0") or go.name.endswith("_1") else go.name
        vis_frames.append(name)

    # Keep track of root path
    root_positions = []

    # Playback timing
    dt = 1.0 / max(args.fps, 1.0)

    for t in range(T):
        rr.set_time_sequence("frame", t)

        row = arr[t]
        q = to_pinocchio_q(row)

        # Forward kinematics with zero velocity is fine for positions/poses
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        # Root pose (world)
        root_tf = data.oMi[1]  # frame 1 is the free-flyer base
        p_root = root_tf.translation
        R_root = R.from_matrix(root_tf.rotation).as_quat()  # (x,y,z,w)

        root_positions.append(p_root.copy())

        # Log root as a transform
        rr.log(
            f"{args.name}/root",
            rr.Transform3D(translation=p_root, rotation=rr.Quaternion(xyzw=R_root))  # Rerun uses xyzw naming
        )

        # Log each visual frame pose relative to world
        for fname in vis_frames:
            fid = model.getFrameId(fname)
            if fid == 0:
                continue
            tf = data.oMf[fid]
            pos = tf.translation
            quat_xyzw = R.from_matrix(tf.rotation).as_quat()
            rr.log(
                f"{args.name}/links/{fname}",
                rr.Transform3D(translation=pos, rotation=rr.Quaternion(xyzw=quat_xyzw))
            )

        # Also log the root path as a growing line strip
        path_np = np.array(root_positions, dtype=float)
        if len(path_np) >= 2:
            rr.log(
                f"{args.name}/root_path",
                rr.LineStrips3D([path_np])  # one strip with all points so far
            )

    print(f"✅ Visualized {T} frames from {os.path.basename(args.csv)} at ~{args.fps} FPS.")
    print("Viewer opened. Scrub the time slider labeled 'frame' to move through the rollout.")


if __name__ == "__main__":
    main()
