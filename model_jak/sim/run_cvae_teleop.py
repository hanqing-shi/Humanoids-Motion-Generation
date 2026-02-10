#!/usr/bin/env python3
"""
MuJoCo teleop using TrajCVAE samples, rendered in MuJoCo.

Key debug-mode fix:
  - DO NOT apply floating base from the model.
  - Apply joints only (qpos[7:36]).
This isolates joint ordering / pose quality without teleporting/rotating the base.

Run:
  python -m model.sim.run_cvae_teleop
  python -m model.sim.run_cvae_teleop --model_path checkpoints/TrajCVAE_best_run.pt
"""

import time
import argparse
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

from model.sim.hid_controller import open_controller, decode_report, joystick_to_base_velocities
from model.sim.motion_generator import load_trajcvae_generator

# =========================
# SIM / CONTROL CONFIG
# =========================

SIM_FPS = 60.0
SIM_DT = 1.0 / SIM_FPS

GEN_HZ = 30.0                 # TrajCVAE playback rate (often 30 Hz)
HORIZON_STEPS = 20            # horizon steps per plan (matches inference_rt default)
REPLAN_INTERVAL = 0.15        # seconds (smaller = more responsive)

AXIS_DEADZONE = 0.10          # consistent with inference_rt-style deadzone
MAX_LIN_VEL = 1.0
MAX_ANG_VEL = 1.0

# MuJoCo model paths
MUJOCO_MENAGERIE_DIR = Path(__file__).resolve().parents[2] / "mujoco_menagerie"
if not MUJOCO_MENAGERIE_DIR.exists():
    MUJOCO_MENAGERIE_DIR = Path(__file__).resolve().parents[3] / "mujoco_menagerie"

G1_XML_PATH = MUJOCO_MENAGERIE_DIR / "unitree_g1" / "g1.xml"
G1_SCENE_XML_PATH = MUJOCO_MENAGERIE_DIR / "unitree_g1" / "scene.xml"


def find_free_joint_id(mj_model: mujoco.MjModel):
    for i in range(mj_model.njnt):
        if mj_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            return i
    return None


def normalize_xyzw_quat_inplace(q36_xyzw: np.ndarray) -> np.ndarray:
    """
    q36_xyzw format: [x y z qx qy qz qw, ...]
    Normalize xyzw quaternion in-place for safety.
    """
    q = q36_xyzw[3:7]
    n = float(np.linalg.norm(q))
    if n > 1e-8:
        q36_xyzw[3:7] = q / n
    else:
        q36_xyzw[3:7] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return q36_xyzw


def apply_joints_only(mj_data: mujoco.MjData, q36_xyzw: np.ndarray):
    """
    q36_xyzw format: [x y z qx qy qz qw, joints(29)]
    MuJoCo qpos layout: [x y z qw qx qy qz, joints(29)]
    For debugging, we keep base fixed and apply ONLY joints (indices 7:36).

    This avoids:
      - teleporting base (x,y,z)
      - feeding non-unit quaternions into MuJoCo
      - base convention mismatch dominating the visual result
    """
    q36_xyzw = np.asarray(q36_xyzw, dtype=np.float32)

    # Preserve base (pos+quat) exactly as MuJoCo currently has it
    base = mj_data.qpos[:7].copy()

    # Apply joint angles ONLY
    # q36_xyzw[7:36] assumed to be 29 joint DoFs
    mj_data.qpos[7:36] = q36_xyzw[7:36]

    # Restore base
    mj_data.qpos[:7] = base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/TrajCVAE_best_run.pt",
        help="TrajCVAE checkpoint path (relative to repo root is fine)."
    )
    parser.add_argument("--mujoco_model", type=str, choices=["scene", "g1"], default="scene")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--past_length", type=int, default=10)
    parser.add_argument("--debug_hid", action="store_true")
    parser.add_argument("--debug_samples", action="store_true", help="Print sample stats once per replan.")
    args = parser.parse_args()

    # --- load MuJoCo XML
    xml_path = (G1_SCENE_XML_PATH if args.mujoco_model == "scene" else G1_XML_PATH)
    if args.mujoco_model == "scene" and not xml_path.exists():
        xml_path = G1_XML_PATH
    if not xml_path.exists():
        raise FileNotFoundError(f"MuJoCo model not found: {xml_path}")

    print(f"Loading MuJoCo model from: {xml_path}")
    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    mj_data = mujoco.MjData(mj_model)
    print(f"Model loaded: nq={mj_model.nq}, nu={mj_model.nu}, njnt={mj_model.njnt}")

    free_joint_id = find_free_joint_id(mj_model)
    if free_joint_id is None:
        print("⚠️  Warning: no free joint found. Camera follow may not work.")
    else:
        print(f"Found free joint (base) at joint index {free_joint_id}")

    # Dummy mode (for now): no gravity
    mj_model.opt.gravity[:] = 0.0

    # --- load TrajCVAE generator
    gen = load_trajcvae_generator(
        ckpt_path=args.model_path,
        device=args.device,
        past_length=args.past_length
    )

    # --- Initialize MuJoCo base + joints to a neutral-ish pose
    # Keep base upright, set some height. Joints start at 0.
    mj_data.qpos[:] = 0.0
    if free_joint_id is not None:
        adr = mj_model.jnt_qposadr[free_joint_id]
        mj_data.qpos[adr + 0] = 0.0  # x
        mj_data.qpos[adr + 1] = 0.0  # y
        mj_data.qpos[adr + 2] = 0.78 # z (reasonable standing height)
        mj_data.qpos[adr + 3] = 1.0  # qw
        mj_data.qpos[adr + 4] = 0.0  # qx
        mj_data.qpos[adr + 5] = 0.0  # qy
        mj_data.qpos[adr + 6] = 0.0  # qz
    mujoco.mj_forward(mj_model, mj_data)
    print("Initialized MuJoCo base to upright standing pose (base fixed; joints will be driven by TrajCVAE).")

    # --- open controller
    dev, profile, _info = open_controller()
    print("\n✅ Controller connected!")
    print("Controls:")
    print("  Left stick Y: forward/back (vx)")
    print("  Left stick X: yaw (wz)")
    print("  Right stick X: left/right (vy)")
    print("  Ctrl+C to quit\n")

    # --- planning / playback buffers
    traj_buf = None            # (T,36) in model format (xyzw)
    traj_step = 0
    last_plan_t = 0.0

    # Sim at 60Hz, generator at 30Hz => hold each generated frame for 2 sim ticks
    hold_ticks = max(1, int(round(SIM_FPS / GEN_HZ)))
    hold_ctr = 0

    last_state = None

    print("Starting MuJoCo viewer...")

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.lookat[:] = [0, 0, 0.8]
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -20

        try:
            while viewer.is_running():
                # ----- read HID
                report = dev.read(64, 0)
                if report:
                    st = decode_report(bytes(report), profile=profile, deadzone=AXIS_DEADZONE, debug=args.debug_hid)
                    if st is not None:
                        last_state = st

                vx, vy, wz = joystick_to_base_velocities(last_state, MAX_LIN_VEL, MAX_ANG_VEL)

                # ----- replan trajectory periodically OR if buffer empty
                now = time.time()
                need_plan = (traj_buf is None) or (traj_step >= len(traj_buf)) or ((now - last_plan_t) >= REPLAN_INTERVAL)

                if need_plan:
                    # Constant command sequence for the horizon (20 steps @ 30Hz)
                    cond_future = np.tile(np.array([vx, vy, wz], dtype=np.float32), (HORIZON_STEPS, 1))  # (20,3)
                    traj_buf = gen.sample(cond_future)  # (20,36) xyzw
                    traj_step = 0
                    last_plan_t = now

                    if args.debug_samples:
                        q0 = traj_buf[0].copy()
                        normalize_xyzw_quat_inplace(q0)
                        quat = q0[3:7]
                        qnorm = float(np.linalg.norm(quat))
                        jmin = float(np.min(q0[7:36]))
                        jmax = float(np.max(q0[7:36]))
                        print(f"DEBUG: traj_buf shape={traj_buf.shape}")
                        print(f"DEBUG: sample[0][:10]={traj_buf[0][:10]}")
                        print(f"DEBUG: quat xyzw={quat} norm={qnorm:.4f} | joints min/max={jmin:+.3f}/{jmax:+.3f} | cmd vx/vy/wz={vx:+.2f}/{vy:+.2f}/{wz:+.2f}")

                # ----- apply trajectory at GEN_HZ with hold
                if traj_buf is not None and traj_step < len(traj_buf):
                    if hold_ctr == 0:
                        q_xyzw = traj_buf[traj_step].copy()
                        normalize_xyzw_quat_inplace(q_xyzw)  # safe (even if we aren't applying base yet)
                        apply_joints_only(mj_data, q_xyzw)
                        traj_step += 1
                        hold_ctr = hold_ticks - 1
                    else:
                        hold_ctr -= 1

                mujoco.mj_forward(mj_model, mj_data)

                # camera follow base
                if free_joint_id is not None:
                    adr = mj_model.jnt_qposadr[free_joint_id]
                    base_pos = mj_data.qpos[adr:adr + 3]
                    viewer.cam.lookat[:] = base_pos.copy()
                    viewer.cam.lookat[2] += 0.8

                viewer.sync()
                time.sleep(SIM_DT)

        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            try:
                dev.close()
            except Exception:
                pass
            print("Controller closed.")


if __name__ == "__main__":
    main()