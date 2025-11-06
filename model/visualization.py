import os
import glob
import time
import numpy as np
import rerun as rr
import trimesh
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

# ========== CONFIG ==========
csv_path = "./results_generated/merged_sample_000_b29_n0.csv"
urdf_path = "./dataset/g1_retargeted_dataset/g1/g1_29dof_rev_1_0.urdf"
urdf_dir = "./dataset/g1_retargeted_dataset/g1"
robot_name = "g1"
frame_rate = 30 
# ============================

# ===== define header =====
frame_names = [
    'pelvis', 'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link', 'left_knee_link',
    'left_ankle_pitch_link', 'left_ankle_roll_link', 'pelvis_contour_link', 'right_hip_pitch_link',
    'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_pitch_link',
    'right_ankle_roll_link', 'waist_yaw_link', 'waist_roll_link', 'torso_link', 'head_link',
    'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link',
    'left_elbow_link', 'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link',
    'left_rubber_hand', 'logo_link', 'right_shoulder_pitch_link', 'right_shoulder_roll_link',
    'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_roll_link', 'right_wrist_pitch_link',
    'right_wrist_yaw_link', 'right_rubber_hand'
]

feature_header = [
    'root_x', 'root_y', 'root_z',
    'root_qw', 'root_qx', 'root_qy', 'root_qz',
    'root_vx', 'root_vy', 'root_vz',
    'root_wx', 'root_wy', 'root_wz'
]
for f in frame_names:
    feature_header += [
        f'{f}_pos_x', f'{f}_pos_y', f'{f}_pos_z',
        f'{f}_ori_w', f'{f}_ori_x', f'{f}_ori_y', f'{f}_ori_z',
        #f'{f}_vel_x', f'{f}_vel_y', f'{f}_vel_z',
        #f'{f}_angvel_x', f'{f}_angvel_y', f'{f}_angvel_z'
    ]
# ==============================================

rr.init("VisualizeCSVMesh_NoHeader", spawn=True)
rr.log("", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

robot = pin.RobotWrapper.BuildFromURDF(urdf_path, urdf_dir, pin.JointModelFreeFlyer())
print(f"✅ Loaded URDF: {urdf_path}")

link2mesh = {}
for visual in robot.visual_model.geometryObjects:
    frame_name = visual.name[:-2]
    mesh = trimesh.load_mesh(visual.meshPath)
    mesh.visual = trimesh.visual.ColorVisuals()
    mesh.visual.vertex_colors = visual.meshColor
    link2mesh[frame_name] = mesh
print(f"✅ Loaded {len(link2mesh)} meshes")

for f, mesh in link2mesh.items():
    rr.log(
        f"{robot_name}/{f}/mesh",
        rr.Mesh3D(
            vertex_positions=mesh.vertices,
            triangle_indices=mesh.faces,
            vertex_normals=mesh.vertex_normals,
            vertex_colors=mesh.visual.vertex_colors,
        ),
        static=True, 
    )

print(f"▶️ Visualizing {csv_path}")
data = np.genfromtxt(csv_path, delimiter=",", names=feature_header)
num_frames = data.shape[0]

# visualize frames
for frame_idx in range(num_frames):
    rr.set_time_sequence("frame", frame_idx)
    
    pos_root = np.array([data["root_x"][frame_idx],
                            data["root_y"][frame_idx],
                            data["root_z"][frame_idx]])
    quat_root = np.array([data["root_qx"][frame_idx],
                            data["root_qy"][frame_idx],
                            data["root_qz"][frame_idx],
                            data["root_qw"][frame_idx]])
    R_root = R.from_quat(quat_root).as_matrix()
    rr.log(f"{robot_name}/root",
            rr.Transform3D(translation=pos_root, mat3x3=R_root, axis_length=0.05))

    for f in frame_names:
        if f not in link2mesh:
            continue

        pos = np.array([
            data[f"{f}_pos_x"][frame_idx],
            data[f"{f}_pos_y"][frame_idx],
            data[f"{f}_pos_z"][frame_idx],
        ])
        quat = np.array([
            data[f"{f}_ori_x"][frame_idx],
            data[f"{f}_ori_y"][frame_idx],
            data[f"{f}_ori_z"][frame_idx],
            data[f"{f}_ori_w"][frame_idx],
        ])
        R_link = R.from_quat(quat).as_matrix()

        pos_world = pos_root + R_root @ pos
        R_world = R_root @ R_link

        rr.log(
            f"{robot_name}/{f}",
            rr.Transform3D(translation=pos_world, mat3x3=R_world),
        )

    time.sleep(1.0 / frame_rate)
