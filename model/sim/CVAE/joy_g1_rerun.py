#!/usr/bin/env python3
import os
import time
import math
from pathlib import Path

import hid
import numpy as np
import pinocchio as pin
import rerun as rr

# ============================================================
# CONFIG
# ============================================================

FPS = 30.0
DT = 1.0 / FPS

CSV_NAME = "walk1_subject1_mid_1.csv"   # pose only
CSV_DIR  = "../retarget_out"            # relative to THIS script

MAX_JOINT_VEL = 0.6   # joystick influence (rad/s)

# Logitech Cordless RumblePad 2
VENDOR_ID  = 0x046D
PRODUCT_ID = 0xC219


# ============================================================
# LOAD CSV + COMPUTE VELOCITY
# ============================================================

def load_csv_traj():
    csv_path = Path(__file__).parent / CSV_DIR / CSV_NAME
    data = np.genfromtxt(csv_path, delimiter=",")
    T, D = data.shape

    vel = np.zeros_like(data)
    vel[1:-1] = (data[2:] - data[:-2]) * (FPS * 0.5)
    vel[0]  = (data[1]  - data[0])  * FPS
    vel[-1] = (data[-1] - data[-2]) * FPS

    print(f"Loaded CSV: {csv_path}")
    print(f"Trajectory shape = {data.shape}")

    return data, vel


# ============================================================
# HID JOYSTICK
# ============================================================

def open_rumblepad():
    dev = hid.device()
    dev.open(VENDOR_ID, PRODUCT_ID)
    dev.set_nonblocking(True)
    return dev

def _norm(v: int) -> float:
    """0~255 -> -1 ~ 1"""
    return (v - 128) / 128.0

def decode(report):
    if not report or len(report) < 7:
        return None

    buttons_low  = report[0]
    buttons_high = report[1]

    buttons = []
    for i in range(8):
        if buttons_low & (1 << i):
            buttons.append(i)
    for i in range(4):
        if buttons_high & (1 << i):
            buttons.append(i + 8)

    lx = _norm(report[2])
    ly = _norm(report[3])
    rx = _norm(report[4])
    ry = _norm(report[5])

    return {
        "axes": dict(lx=lx, ly=ly, rx=rx, ry=ry),
        "buttons": buttons,
    }


# ============================================================
# LOAD UNITREE G1 (Pinocchio)
# ============================================================

def load_g1_robot():
    base = Path(__file__).resolve().parents[1]  # motion_generation_swv1/
    urdf = base / "robot_description" / "g1" / "g1_29dof_rev_1_0.urdf"

    robot = pin.RobotWrapper.BuildFromURDF(
        str(urdf),
        str(urdf.parent),
        pin.JointModelFreeFlyer(),
    )
    print("Loaded URDF:", urdf)
    print("nq =", robot.model.nq)
    return robot


# ============================================================
# Rerun helper for G1 (skeleton points)
# ============================================================

class RerunG1:
    def __init__(self, robot: pin.RobotWrapper):
        self.robot = robot
        self.model = robot.model
        self.data = robot.data

        # 한 번 프레임 이름들 보여주기 (joint index 참고용)
        print("=== Frames (for debug) ===")
        for i, fr in enumerate(self.model.frames):
            print(i, fr.name)

    def update(self, q: np.ndarray):
        """
        q: (nq,) configuration
        Pinocchio FK -> 모든 frame 위치를 Points3D로 로그
        """
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        positions = []
        for i, fr in enumerate(self.model.frames):
            T = self.data.oMf[i]
            positions.append(T.translation)

        positions = np.array(positions, dtype=np.float32)

        rr.log(
            "g1/frames",
            rr.Points3D(
                positions=positions,
                radii=0.01,
            ),
        )


# ============================================================
# JOYSTICK → JOINT VELOCITIES
# ============================================================

# ⚠️ 일단 대충 넣어둔 값. 나중에 실제 joint index로 바꿀 것.
JOINT_MAP = {
    "l_shoulder_pitch": 10,
    "l_elbow":          12,
    "r_shoulder_pitch": 17,
    "r_elbow":          19,
}

def joystick_qdot(decoded, nq: int) -> np.ndarray:
    qdot = np.zeros(nq, dtype=np.float32)
    if decoded is None:
        return qdot

    axes = decoded["axes"]

    if "l_shoulder_pitch" in JOINT_MAP and JOINT_MAP["l_shoulder_pitch"] < nq:
        qdot[JOINT_MAP["l_shoulder_pitch"]] = -axes["ly"] * MAX_JOINT_VEL
    if "l_elbow" in JOINT_MAP and JOINT_MAP["l_elbow"] < nq:
        qdot[JOINT_MAP["l_elbow"]] = axes["lx"] * MAX_JOINT_VEL

    if "r_shoulder_pitch" in JOINT_MAP and JOINT_MAP["r_shoulder_pitch"] < nq:
        qdot[JOINT_MAP["r_shoulder_pitch"]] = -axes["ry"] * MAX_JOINT_VEL
    if "r_elbow" in JOINT_MAP and JOINT_MAP["r_elbow"] < nq:
        qdot[JOINT_MAP["r_elbow"]] = axes["rx"] * MAX_JOINT_VEL

    return qdot


# ============================================================
# RERUN INIT
# ============================================================

def init_rerun():
    rr.init("G1 CSV + Joystick", spawn=True)
    rr.log("", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    # 1) Trajectory + velocity from CSV
    q_traj, vel_traj = load_csv_traj()
    T, D = q_traj.shape

    # 2) G1 robot
    robot = load_g1_robot()
    nq = robot.model.nq

    if D != nq:
        print(f"⚠ WARNING: CSV dim {D} != robot.nq {nq}. Using min().")
    dim = min(D, nq)

    # 3) Rerun skeleton renderer
    init_rerun()
    rr_g1 = RerunG1(robot)

    # 4) current state
    idx = 0
    q = q_traj[0].copy()
    q = q.astype(np.float32)

    # 5) Joystick
    dev = open_rumblepad()
    print("✅ RumblePad connected. Move sticks, press BUTTON 0 to reset. Ctrl+C to quit.")

    frame = 0

    try:
        while True:
            report = dev.read(64)
            decoded = decode(report)

            # reset to first frame
            if decoded and 0 in decoded["buttons"]:
                print("Reset to first CSV frame.")
                idx = 0
                q[:] = q_traj[0]

            # CSV 기반 vel
            vel_csv = vel_traj[idx]

            # joystick 기반 추가 vel
            qdot_joy = joystick_qdot(decoded, dim)

            # 통합 속도 (앞 dim 개만 쓴다)
            q[:dim] = q[:dim] + (vel_csv[:dim] + qdot_joy[:dim]) * DT

            # Rerun에 로깅
            rr.set_time_sequence("frame", frame)
            rr_g1.update(q)

            # 다음 프레임
            idx = (idx + 1) % T
            frame += 1
            time.sleep(DT)

    except KeyboardInterrupt:
        print("\n[joy_g1_rerun] 종료합니다.")
    finally:
        dev.close()


if __name__ == "__main__":
    main()
