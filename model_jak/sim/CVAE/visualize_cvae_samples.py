#!/usr/bin/env python3
import numpy as np
import torch
import rerun as rr
from train_motion_cvae import MotionCVAE, sample_future, COND_DIM, FUTURE_DIM, LATENT_DIM, D_CONF
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "motion_cvae.pt")
CSV_PATH = os.path.join(BASE_DIR, "retarget_out_vel", "walk1_subject1_mid_vel.csv")

# -----------------------------
# CONFIG
# -----------------------------

NUM_SAMPLES = 3   # 몇 개의 latent z를 샘플링할지
FPS = 30          # 1초 30프레임 기준

# -----------------------------
# LOAD MODEL
# -----------------------------
ckpt = torch.load(MODEL_PATH, map_location="cpu")
model = MotionCVAE(
    cond_dim=COND_DIM,
    future_dim=FUTURE_DIM,
    latent_dim=ckpt["latent_dim"]
)
model.load_state_dict(ckpt["model_state"])

# -----------------------------
# LOAD CONDITION
# -----------------------------
data = np.genfromtxt(CSV_PATH, delimiter=',', skip_header=1)
q_t = data[0, :D_CONF]
vel_t = data[0, D_CONF:D_CONF+3]
cond_vec = np.concatenate([q_t, vel_t])

# -----------------------------
# SAMPLE FUTURES
# -----------------------------
samples = sample_future(model, cond_vec, n_samples=NUM_SAMPLES)

# -----------------------------
# RERUN VISUALIZATION
# -----------------------------
rr.init("CVAE Motion Samples", spawn=True)
rr.log("", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

# 각 trajectory를 색깔 다르게 시각화
colors = [
    (255, 0, 0),     # 빨강
    (0, 255, 0),     # 초록
    (0, 128, 255),   # 파랑
]

for i, traj in enumerate(samples):
    # traj shape: (30, 36)
    name = f"trajectory_{i}"
    color = colors[i % len(colors)]
    for frame_idx in range(traj.shape[0]):
        rr.set_time_sequence("frame", frame_idx)
        q = traj[frame_idx, :]
        # 단순히 base (x, y, z) 위치를 궤적으로 찍는다.
        # 실제 full-body mesh를 보고 싶다면 rerun_urdf 활용 가능.
        base_xyz = q[:3]
        rr.log(
            f"{name}/base",
            rr.Points3D(
                positions=np.array([base_xyz]),
                colors=np.array([color]),
                radii=0.02,
            ),
        )

