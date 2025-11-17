import os
import time
import argparse
import numpy as np
import torch

from dataset import MotionDataset
from models import TrajCVAE

import rerun as rr
import trimesh
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

# Optional joystick support
try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

DEFAULT_DATA_DIR  = "./dataset/data_feature"
DEFAULT_LABEL_DIR = "./dataset/data_label"
DEFAULT_MOTIONS   = ["walk"]
DEFAULT_CKPT      = "./checkpoints/TrajCVAE_best.pt"

PAST = 10      # must match past_lenth in training pipeline
H    = 20      # horizon per step

# joystick scaling
MAX_VX = 1.0   # forward/back
MAX_VY = 1.0   # strafe
MAX_WZ = 1.0   # yaw rate

FPS = 30
DT  = 1.0 / FPS

URDF_PATH = "./dataset/g1_retargeted_dataset/g1/g1_29dof_rev_1_0.urdf"
URDF_DIR  = "./dataset/g1_retargeted_dataset/g1"
ROBOT_NAME = "g1"

FRAME_NAMES = [
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


def init_joystick():
    if not HAS_PYGAME:
        print("⚠️ pygame not installed; joystick disabled.")
        return None

    pygame.init()
    pygame.joystick.init()
    js = None
    if pygame.joystick.get_count() > 0:
        js = pygame.joystick.Joystick(0)
        js.init()
        print(f"🎮 Using joystick: {js.get_name()}")
    else:
        print("⚠️ No joystick detected; commands will be zero.")
    return js


def read_joystick_command(js):
    """
    Return [linear_x, linear_y, angular_z] from joystick.

    Mapping (what you requested):
      - Left stick Y (axis 1): forward/back    -> linear_x (invert so up = +)
      - Right stick X (axis 3): strafe        -> linear_y
      - Left stick X (axis 0): turning        -> angular_z
    """
    if js is None:
        return np.zeros(3, dtype=np.float32)

    pygame.event.pump()

    left_x  = js.get_axis(0)   # left stick horizontal
    left_y  = js.get_axis(1)   # left stick vertical
    right_x = js.get_axis(3) if js.get_numaxes() > 3 else js.get_axis(2)

    linear_x  = -left_y * MAX_VX   # up on stick = forward
    linear_y  =  right_x * MAX_VY  # right = strafe right
    angular_z =  left_x * MAX_WZ   # right = positive yaw

    return np.array([linear_x, linear_y, angular_z], dtype=np.float32)


def to_torch(x, device):
    return torch.tensor(x, dtype=torch.float32, device=device)


def load_robot_and_meshes():
    robot = pin.RobotWrapper.BuildFromURDF(URDF_PATH, URDF_DIR, pin.JointModelFreeFlyer())
    print(f"✅ Loaded URDF: {URDF_PATH}")

    link2mesh = {}
    for visual in robot.visual_model.geometryObjects:
        frame_name = visual.name[:-2]
        mesh = trimesh.load_mesh(visual.meshPath)
        mesh.visual = trimesh.visual.ColorVisuals()
        mesh.visual.vertex_colors = visual.meshColor
        link2mesh[frame_name] = mesh

    print(f"✅ Loaded {len(link2mesh)} meshes")

    # Log meshes once as static
    for f, mesh in link2mesh.items():
        rr.log(
            f"{ROBOT_NAME}/{f}/mesh",
            rr.Mesh3D(
                vertex_positions=mesh.vertices,
                triangle_indices=mesh.faces,
                vertex_normals=mesh.vertex_normals,
                vertex_colors=mesh.visual.vertex_colors,
            ),
            static=True,
        )
    return robot, link2mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  default=DEFAULT_DATA_DIR)
    parser.add_argument("--label_dir", default=DEFAULT_LABEL_DIR)
    parser.add_argument("--motions",   nargs="+", default=DEFAULT_MOTIONS)
    parser.add_argument("--ckpt",      default=DEFAULT_CKPT)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ds = MotionDataset(
        data_dir=args.data_dir,
        label_dir=args.label_dir,
        seq_len=30,
        motions=args.motions,
        columns=("pos", "ori"),
        stride=10,
        transform=None,
    )
    state_seq, label_seq = ds[0]
    frame_size, traj_dim = state_seq.shape
    cond_dim = label_seq.shape[-1]
    print(f"traj_dim = {traj_dim}, cond_dim = {cond_dim}")

    selected_cols = ds.selected_cols
    name_to_idx = {name: i for i, name in enumerate(selected_cols)}

    # initialize history with real data
    x_hist = state_seq[:PAST].numpy().copy()   # (PAST, traj_dim)
    c_hist = label_seq[:PAST].numpy().copy()   # (PAST, cond_dim)

    # load trained TrajCVAE
    model = TrajCVAE(
        traj_dim=traj_dim,
        cond_dim=cond_dim,
        teacher_forcing=0.0,
        past_lenth=PAST,
    ).to(device)

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"✅ Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Rerun setup
    rr.init("Joystick_G1_Visualizer", spawn=True)
    rr.log("", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    robot, link2mesh = load_robot_and_meshes()

    js = init_joystick()
    u_prev = np.zeros(3, dtype=np.float32)

    print("Starting joystick sim. Ctrl+C to quit.")
    start_time = time.time()
    t_prev = start_time
    frame_idx = 0

    try:
        while True:
            u_raw = read_joystick_command(js)
            u = 0.8 * u_prev + 0.2 * u_raw
            u_prev = u

            x_past = x_hist[None, :, :]        # (1, PAST, traj_dim)
            cond_past = c_hist[None, :, :]     # (1, PAST, cond_dim)

            cond_future = np.repeat(u[None, :], H, axis=0)   # (H, cond_dim)
            cond_future = cond_future[None, :, :]            # (1, H, cond_dim)

            x0 = x_hist[-1]                                   # last state
            x_future = np.repeat(x0[None, :], H, axis=0)      # seed, (H, traj_dim)
            x_future = x_future[None, :, :]                   # (1, H, traj_dim)

            with torch.no_grad():
                x_past_t      = to_torch(x_past, device)
                cond_past_t   = to_torch(cond_past, device)
                cond_future_t = to_torch(cond_future, device)
                x_future_t    = to_torch(x_future, device)

                y_future = model.sample(cond_past_t, cond_future_t, x_past_t, x_future_t)
                # y_future: (1, H, traj_dim) index 1 = first new frame after x0
                x_next = y_future[0, 1].cpu().numpy()  # (traj_dim,)

            # update history frames
            x_hist = np.concatenate([x_hist[1:], x_next[None, :]], axis=0)
            c_hist = np.concatenate([c_hist[1:], u[None, :]], axis=0)

            # decode feature vector
            rr.set_time(timestamp=time.time())
            # root pose
            ix_rx = name_to_idx["root_x"]
            ix_ry = name_to_idx["root_y"]
            ix_rz = name_to_idx["root_z"]
            ix_qw = name_to_idx["root_qw"]
            ix_qx = name_to_idx["root_qx"]
            ix_qy = name_to_idx["root_qy"]
            ix_qz = name_to_idx["root_qz"]

            pos_root = np.array([x_next[ix_rx], x_next[ix_ry], x_next[ix_rz]])
            # R.from_quat expects [x, y, z, w]
            quat_root = np.array([
                x_next[ix_qx],
                x_next[ix_qy],
                x_next[ix_qz],
                x_next[ix_qw],
            ])
            R_root = R.from_quat(quat_root).as_matrix()
            rr.log(
                f"{ROBOT_NAME}/root",
                rr.Transform3D(translation=pos_root, mat3x3=R_root, axis_length=0.05),
            )

            for f in FRAME_NAMES:
                if f not in link2mesh:
                    continue
                px = name_to_idx.get(f"{f}_pos_x")
                py = name_to_idx.get(f"{f}_pos_y")
                pz = name_to_idx.get(f"{f}_pos_z")
                ox = name_to_idx.get(f"{f}_ori_x")
                oy = name_to_idx.get(f"{f}_ori_y")
                oz = name_to_idx.get(f"{f}_ori_z")
                ow = name_to_idx.get(f"{f}_ori_w")
                if None in (px, py, pz, ox, oy, oz, ow):
                    continue

                pos_local = np.array([x_next[px], x_next[py], x_next[pz]])
                quat_local = np.array([
                    x_next[ox],
                    x_next[oy],
                    x_next[oz],
                    x_next[ow],
                ])
                R_link = R.from_quat(quat_local).as_matrix()

                pos_world = pos_root + R_root @ pos_local
                R_world = R_root @ R_link

                rr.log(
                    f"{ROBOT_NAME}/{f}",
                    rr.Transform3D(translation=pos_world, mat3x3=R_world),
                )

            frame_idx += 1

            t_now = time.time()
            sleep = DT - (t_now - t_prev)
            if sleep > 0:
                time.sleep(sleep)
            t_prev = time.time()

    except KeyboardInterrupt:
        print("\nExiting joystick sim...")
    finally:
        if HAS_PYGAME:
            pygame.quit()

if __name__ == "__main__":
    main()

