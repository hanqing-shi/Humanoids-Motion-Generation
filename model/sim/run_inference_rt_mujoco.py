#!/usr/bin/env python3
"""
MuJoCo visualization of inference_rt pipeline.

This file mirrors inference_rt.py logic (joystick -> model.sample -> safety -> tracker -> closed-loop history),
but visualizes in MuJoCo (unitree_g1 menagerie scene) instead of Rerun.

Run from repo root:
  python -m model.sim.run_inference_rt_mujoco

Optional:
  python -m model.sim.run_inference_rt_mujoco --ckpt checkpoints/TrajCVAE_best_run.pt
"""

import os
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import mujoco
import mujoco.viewer


# ----------------------------
# Robust imports (repo layout)
# ----------------------------
# models
try:
    from model import models
except Exception:
    import models  # fallback (older layout)

# joystick
try:
    from model.joystick import JoystickController
except Exception:
    try:
        from joystick import JoystickController
    except Exception as e:
        raise ImportError("Could not import JoystickController. Check your repo paths.") from e

# safety + tracker + diagnostics
try:
    from model.control.g1.safety import G1SafetyFilter
    from model.control.g1.tracking.tracker import HybridPDTracker, HybridPDParams
    from model.control.g1.limits import get_joint_limits
except Exception:
    # fallback to old relative imports
    from control.g1.safety import G1SafetyFilter
    from control.g1.tracking.tracker import HybridPDTracker, HybridPDParams
    from control.g1.limits import get_joint_limits

USE_DIAGNOSTICS = True
if USE_DIAGNOSTICS:
    try:
        from model.control.g1.diagnostics import split_actuated, set_actuated, assert_full_configuration
    except Exception:
        try:
            from control.g1.diagnostics import split_actuated, set_actuated, assert_full_configuration
        except Exception:
            USE_DIAGNOSTICS = False


# ----------------------------
# CLI
# ----------------------------
def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TrajCVAE", help="Model class name in models.py")
    parser.add_argument("--past-lenth", type=int, default=10)
    parser.add_argument("--inference-step", type=int, default=300)
    parser.add_argument("--hz", type=float, default=30.0, help="Control/vis rate (Hz)")
    parser.add_argument("--chunk", type=int, default=20, help="Steps sampled per outer frame")

    # MuJoCo / assets
    parser.add_argument("--mujoco_model", choices=["scene", "g1"], default="scene")
    parser.add_argument("--menagerie_dir", type=str, default="", help="Optional override path to mujoco_menagerie")

    # checkpoint override
    parser.add_argument("--ckpt", type=str, default="", help="Optional checkpoint path override")
    return parser.parse_args()


# ----------------------------
# Helpers
# ----------------------------
def _quat_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    """Convert [x,y,z,w] -> [w,x,y,z]."""
    x, y, z, w = q_xyzw
    return np.array([w, x, y, z], dtype=np.float32)


def _find_free_joint_id(m: mujoco.MjModel):
    for i in range(m.njnt):
        if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            return i
    return None


def _build_actuated_qpos_index_map(mj_model: mujoco.MjModel):
    """
    Build a mapping from your 29 actuated joints (Pinocchio q[7:36] order)
    to MuJoCo qpos indices, using joint names from get_joint_limits().

    Returns:
      qpos_indices: list[int] length 29
      names_actuated: list[str] length 29
    """
    _, _, names_actuated = get_joint_limits()  # names for q[7]..q[35] in your convention

    qpos_indices = []
    missing = []

    for name in names_actuated:
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)

        # try common variants
        if jid < 0 and name.endswith("_joint"):
            jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name[:-6])
        if jid < 0 and not name.endswith("_joint"):
            jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name + "_joint")

        if jid < 0:
            missing.append(name)
            qpos_indices.append(None)
            continue

        adr = int(mj_model.jnt_qposadr[jid])  # for hinge joints, this is the qpos slot
        qpos_indices.append(adr)

    if missing:
        print("\n⚠️ Some actuated joint names were not found in MuJoCo model:")
        for n in missing[:20]:
            print("  -", n)
        if len(missing) > 20:
            print(f"  ... +{len(missing)-20} more")
        print("\nIf the robot looks wrong, this is the FIRST thing to fix.")
        print("We need to reconcile joint naming between your URDF/Pinocchio and menagerie MJCF.\n")

    return qpos_indices, names_actuated


def _apply_full_configuration_to_mujoco(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    config36: np.ndarray,
    qpos_indices_29: list,
):
    """
    config36 convention (from inference_rt):
      [0:3]   base pos xyz
      [3:7]   base quat xyzw
      [7:36]  29 actuated joints (Pinocchio order)

    MuJoCo qpos:
      [0:3] pos
      [3:7] quat wxyz
      hinge joints scattered at qpos indices given by qpos_indices_29
    """
    assert config36.shape == (36,), f"expected (36,), got {config36.shape}"

    # base
    mj_data.qpos[0:3] = config36[0:3]
    mj_data.qpos[3:7] = _quat_xyzw_to_wxyz(config36[3:7])

    # joints
    q_act = config36[7:36]
    for i, adr in enumerate(qpos_indices_29):
        if adr is None:
            continue
        mj_data.qpos[adr] = float(q_act[i])


def _init_standing_pose(mj_model, mj_data, x_init_np: np.ndarray, qpos_indices_29: list):
    """Initialize MuJoCo to the same 'x_init_np' pose used by inference_rt."""
    mj_data.qpos[:] = 0.0
    mj_data.qvel[:] = 0.0

    _apply_full_configuration_to_mujoco(mj_model, mj_data, x_init_np, qpos_indices_29)
    mujoco.mj_forward(mj_model, mj_data)


# ----------------------------
# Main
# ----------------------------
def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # ---------------- MuJoCo load ----------------
    if args.menagerie_dir:
        menagerie = Path(args.menagerie_dir)
    else:
        # search up from this file (model/sim/...)
        menagerie = Path(__file__).resolve().parents[2] / "mujoco_menagerie"
        if not menagerie.exists():
            menagerie = Path(__file__).resolve().parents[3] / "mujoco_menagerie"

    g1_xml = menagerie / "unitree_g1" / "g1.xml"
    scene_xml = menagerie / "unitree_g1" / "scene.xml"
    xml_path = scene_xml if (args.mujoco_model == "scene" and scene_xml.exists()) else g1_xml

    if not xml_path.exists():
        raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

    print(f"Loading MuJoCo model from: {xml_path}")
    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    mj_data = mujoco.MjData(mj_model)
    print(f"Model loaded: nq={mj_model.nq}, nu={mj_model.nu}, njnt={mj_model.njnt}")

    free_joint_id = _find_free_joint_id(mj_model)
    if free_joint_id is None:
        print("⚠️ No free joint found. (Unexpected for G1 menagerie)")
    else:
        print(f"Found free joint (base) at joint index {free_joint_id}")

    # you’ve been running “dummy” (no gravity); keep that consistent for now
    mj_model.opt.gravity[:] = 0.0

    # build mapping once
    qpos_indices_29, names_actuated = _build_actuated_qpos_index_map(mj_model)

    # ---------------- Model load (same as inference_rt) ----------------
    ModelClass = getattr(models, args.model)

    if args.model == "TrajCVAE":
        model = ModelClass(traj_dim=36, cond_dim=3, teacher_forcing=0, past_lenth=args.past_lenth).to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    print(f"Loaded model class: {args.model}")

    ckpt_path = args.ckpt if args.ckpt else f"checkpoints/{args.model}_best_walk.pt"
    if not os.path.exists(ckpt_path):
        print(f"⚠️ Warning: checkpoint not found: {ckpt_path}")
    else:
        ckpt = torch.load(ckpt_path, map_location=device)
        # inference_rt expects ckpt["model"]
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            # fallback: maybe it’s already a state_dict
            model.load_state_dict(ckpt)
        print(f"✅ Model weights loaded from: {ckpt_path}")

    model.eval()

    # ---------------- x_init (exact from inference_rt) ----------------
    x_init_np = np.array([
        0.0, 0.0, 0.76,
        0.0, 0.0, 0.0, 1.0,
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        0.0, 0.0, 0.0,
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0
    ], dtype=np.float32)

    if USE_DIAGNOSTICS:
        assert_full_configuration(x_init_np, "x_init_np")

    # init MuJoCo to standing pose
    _init_standing_pose(mj_model, mj_data, x_init_np, qpos_indices_29)
    print("Initialized MuJoCo to inference_rt standing pose.")

    # ---------------- Joystick (same as inference_rt) ----------------
    controller = JoystickController(motion="walk", deadzone=0.1)
    assert controller.joystick is not None, "Joystick not connected. Please connect and retry."
    print("✅ Joystick connected (inference_rt pipeline).")

    hz = float(args.hz)
    dt = 1.0 / hz
    chunk = int(args.chunk)

    # ---------------- Safety + Tracker (same as inference_rt) ----------------
    safety = G1SafetyFilter(dt=dt)

    if USE_DIAGNOSTICS:
        q_init = split_actuated(x_init_np)
    else:
        q_init = x_init_np[-29:]
    safety.reset(q_init)

    params = HybridPDParams(
        alpha=0.4,
        dq_limit=1e6,
        ddq_limit=None,
        kp_default=60.0,
        kd_default=6.0,
        use_critical_damping=True,
        output_dq_cmd=True,
    )
    tracker = HybridPDTracker(n=29, dt=dt, params=params)
    tracker.reset(q_init)

    # ---------------- Inference loop (same structure) ----------------
    x_init = torch.from_numpy(x_init_np).to(device).unsqueeze(0).unsqueeze(0)  # (1,1,36)
    cond_past = torch.zeros((1, args.past_lenth, 3), device=device)
    x_past = x_init.repeat(1, args.past_lenth, 1)
    x_start = x_init[:, -1:, :]

    print("\nStarting MuJoCo viewer...")
    with torch.no_grad(), mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        # camera
        viewer.cam.lookat[:] = [0, 0, 0.8]
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -20

        last_applied_config = None

        try:
            for frame in range(args.inference_step):
                # joystick -> commands (body-frame), same call as inference_rt
                cmd = controller.get_cond_commands(steps=chunk, freq=hz).reshape(1, -1, 3)  # (1,chunk,3)
                cond = torch.from_numpy(cmd).float().to(device)

                # sample
                samples = model.sample(cond_past, cond, x_past, x_start)  # (1,chunk,36) expected

                # step through chunk
                for step in range(samples.shape[1]):
                    configuration = samples[-1, step, :].detach().cpu().numpy().astype(np.float32)  # (36,)

                    # actuated slice
                    if USE_DIAGNOSTICS:
                        q_model = split_actuated(configuration)
                    else:
                        q_model = configuration[-29:]

                    # safety
                    q_safe, info = safety.step(q_model)
                    if not info.get("ok", True):
                        print("⚠️ Safety triggered:", info.get("reason", "unknown"))

                    # tracker uses previous cmd as "measurement" (same as inference_rt)
                    if tracker.q_cmd_prev is None:
                        q_meas = q_init.copy()
                        dq_meas = np.zeros_like(q_meas)
                    else:
                        q_meas = tracker.q_cmd_prev.copy()
                        dq_meas = tracker.dq_cmd_prev.copy()

                    q_cmd, dq_cmd, kp, kd, tau_ff = tracker.step(q_safe, q_meas, dq_meas)

                    # rebuild full configuration with tracked joints
                    if USE_DIAGNOSTICS:
                        configuration_safe = set_actuated(configuration, q_cmd)
                    else:
                        configuration_safe = configuration.copy()
                        configuration_safe[-29:] = q_cmd

                    last_applied_config = configuration_safe

                    # ---- MuJoCo apply ----
                    _apply_full_configuration_to_mujoco(mj_model, mj_data, configuration_safe, qpos_indices_29)
                    mujoco.mj_forward(mj_model, mj_data)

                    # follow base
                    base_pos = mj_data.qpos[0:3]
                    viewer.cam.lookat[:] = base_pos.copy()
                    viewer.cam.lookat[2] += 0.8

                    viewer.sync()
                    time.sleep(dt)

                # closed-loop update (same as inference_rt)
                cond_past = torch.cat([cond_past, cond], dim=1)[:, -args.past_lenth:, :]
                assert last_applied_config is not None, "No applied configuration recorded"
                applied_t = torch.from_numpy(last_applied_config).to(device).view(1, 1, -1)
                x_past = torch.cat([x_past, applied_t], dim=1)[:, -args.past_lenth:, :]
                x_start = applied_t

        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            controller.close()
            print("Controller closed.")


if __name__ == "__main__":
    args = parse_cli()
    main(args)
