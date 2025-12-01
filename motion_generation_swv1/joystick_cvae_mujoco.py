#!/usr/bin/env python3
"""
Note: This script requires both PyTorch and MuJoCo.
Run with: python joystick_cvae_mujoco.py --model_path motion_cvae.pt
(Not mjpython, as mjpython environment may not have PyTorch)
"""
"""
Joystick-controlled CVAE motion generation in MuJoCo.
- Reads joystick input (vx, vy, wz)
- Uses current robot state + joystick command as condition for CVAE
- Generates future motion trajectory
- Plays back in MuJoCo
"""

import os
import time
import math
import argparse
from pathlib import Path

import hid
import numpy as np
import torch
import mujoco
import mujoco.viewer

# ============================================================
# CONFIG
# ============================================================

FPS = 60.0
DT = 1.0 / FPS

# CVAE model parameters (must match training)
HORIZON = 30  # future frames to predict
D_CONF = 36   # joint configuration dimension (nq)
COND_DIM = D_CONF + 3  # [q_t (36), vx, vy, wz (3)] = 39
LATENT_DIM = 32
FUTURE_DIM = HORIZON * D_CONF  # 30 * 36 = 1080

# Joystick control parameters
MAX_LIN_VEL = 1.0  # m/s
MAX_ANG_VEL = 1.0  # rad/s
AXIS_DEADZONE = 0.05

# Logitech F710
VENDOR_ID = 0x046D
PRODUCT_ID = 0xC219

# Paths
MUJOCO_MENAGERIE_DIR = Path(__file__).resolve().parents[2] / "mujoco_menagerie"
G1_XML_PATH = MUJOCO_MENAGERIE_DIR / "unitree_g1" / "g1.xml"
G1_SCENE_XML_PATH = MUJOCO_MENAGERIE_DIR / "unitree_g1" / "scene.xml"


# ============================================================
# CVAE MODEL
# ============================================================

class MotionCVAE(torch.nn.Module):
    """
    CVAE model (must match training architecture).
    - cond: [q_t (36), vx, vy, wz] -> (39,)
    - future: flattened future rollout (30*36,)
    """
    def __init__(self,
                 cond_dim=COND_DIM,
                 future_dim=FUTURE_DIM,
                 latent_dim=LATENT_DIM,
                 hidden_dim=512,
                 depth=4):
        super().__init__()
        self.cond_dim = cond_dim
        self.future_dim = future_dim
        self.latent_dim = latent_dim
        self.horizon = HORIZON
        self.d_conf = D_CONF

        # Encoder
        enc_in_dim = cond_dim + future_dim
        enc_layers = []
        dim = enc_in_dim
        for _ in range(depth):
            enc_layers.append(torch.nn.Linear(dim, hidden_dim))
            enc_layers.append(torch.nn.ReLU())
            dim = hidden_dim
        self.encoder_body = torch.nn.Sequential(*enc_layers)
        self.to_mu = torch.nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = torch.nn.Linear(hidden_dim, latent_dim)

        # Decoder
        dec_in_dim = cond_dim + latent_dim
        dec_layers = []
        dim = dec_in_dim
        for _ in range(depth):
            dec_layers.append(torch.nn.Linear(dim, hidden_dim))
            dec_layers.append(torch.nn.ReLU())
            dim = hidden_dim
        dec_layers.append(torch.nn.Linear(hidden_dim, future_dim))
        self.decoder_body = torch.nn.Sequential(*dec_layers)

    def decode(self, cond, z):
        """
        Args:
            cond: (B, 39) - [q_t (36), vx, vy, wz (3)]
            z: (B, latent_dim)
        Returns:
            future: (B, HORIZON, D_CONF) - future joint configurations
        """
        d_in = torch.cat([cond, z], dim=-1)  # (B, 39+latent)
        flat = self.decoder_body(d_in)  # (B, 1080)
        seq = flat.view(-1, self.horizon, self.d_conf)  # (B, 30, 36)
        return seq


# ============================================================
# JOYSTICK HANDLING
# ============================================================

def find_logitech_gamepad():
    """Find Logitech gamepad by enumerating HID devices."""
    for device_info in hid.enumerate():
        vid = device_info.get("vendor_id", 0)
        pid = device_info.get("product_id", 0)
        name = device_info.get("product_string", "")
        
        if vid == 0x046D:
            if "gamepad" in name.lower() or "f710" in name.lower() or "rumblepad" in name.lower() or pid in [0xC219, 0xC21F, 0xC216]:
                return vid, pid, name
    
    return None, None, None

def open_rumblepad():
    """Open the Logitech gamepad."""
    vid, pid, name = find_logitech_gamepad()
    
    if vid is None:
        vid = VENDOR_ID
        pid = PRODUCT_ID
        print(f"⚠️  Could not auto-detect gamepad, trying VID={vid:04X}, PID={pid:04X}")
    else:
        print(f"✅ Found Logitech gamepad: {name} (VID={vid:04X}, PID={pid:04X})")
    
    dev = hid.device()
    dev.open(vid, pid)
    dev.set_nonblocking(True)
    return dev

def _norm(v: int) -> float:
    """Normalize 0~255 -> -1.0~+1.0."""
    return (v - 128) / 128.0

def decode_joystick(report):
    """Decode HID report from F710."""
    if not report or len(report) < 7:
        return None

    buttons_low = report[0]
    buttons_high = report[1]

    buttons = []
    for i in range(8):
        if buttons_low & (1 << i):
            buttons.append(i)
    for i in range(4):
        if buttons_high & (1 << i):
            buttons.append(i + 8)

    # F710 mapping: byte_1, byte_2, byte_3, byte_4 = axis0, axis1, axis2, axis3
    axis0 = _norm(report[1]) if len(report) > 1 else 0.0  # LEFT STICK X
    axis1 = _norm(report[2]) if len(report) > 2 else 0.0  # LEFT STICK Y
    axis2 = _norm(report[3]) if len(report) > 3 else 0.0  # RIGHT STICK X
    axis3 = _norm(report[4]) if len(report) > 4 else 0.0  # RIGHT STICK Y

    # Apply deadzone
    def apply_deadzone(val, deadzone):
        if abs(val) < deadzone:
            return 0.0
        return val

    axis0 = apply_deadzone(axis0, AXIS_DEADZONE)
    axis1 = apply_deadzone(axis1, AXIS_DEADZONE)
    axis2 = apply_deadzone(axis2, AXIS_DEADZONE)
    axis3 = apply_deadzone(axis3, AXIS_DEADZONE)

    return {
        "axes": dict(
            axis0=axis0, axis1=axis1, axis2=axis2, axis3=axis3,
            lx=axis0, ly=axis1, rx=axis2, ry=axis3
        ),
        "buttons": buttons,
    }

def joystick_to_base_velocities(decoded):
    """
    Extract base velocities from joystick.
    Returns: (vx, vy, omega_z)
    - vx: forward/backward velocity (m/s) - L STICK AXIS 1 (Y, vertical)
    - vy: left/right velocity (m/s) - R STICK AXIS 2
    - omega_z: yaw angular velocity (rad/s) - L STICK AXIS 0 (X, horizontal)
    """
    if decoded is None:
        return 0.0, 0.0, 0.0
    
    axes = decoded["axes"]
    
    # vx: forward/backward - L STICK AXIS 1 (Y, vertical) - up = forward, down = backward
    vx = -axes.get("axis1", axes.get("ly", 0.0)) * MAX_LIN_VEL
    
    # vy: left/right - R STICK AXIS 2 (left = positive, right = negative) - reversed
    vy = -axes.get("axis2", axes.get("rx", 0.0)) * MAX_LIN_VEL
    
    # omega_z: yaw rotation - L STICK AXIS 0 (X, horizontal) - left = positive, right = negative - reversed
    omega_z = -axes.get("axis0", axes.get("lx", 0.0)) * MAX_ANG_VEL
    
    return vx, vy, omega_z


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Joystick-controlled CVAE motion generation in MuJoCo")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained CVAE model (.pt file)"
    )
    parser.add_argument(
        "--mujoco_model",
        type=str,
        choices=["scene", "g1"],
        default="scene",
        help="MuJoCo model: 'scene' (with grid) or 'g1' (plain)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for model inference: 'cpu' or 'cuda'"
    )
    
    args = parser.parse_args()
    
    # Load MuJoCo model
    if args.mujoco_model == "scene":
        xml_path = G1_SCENE_XML_PATH if G1_SCENE_XML_PATH.exists() else G1_XML_PATH
    else:
        xml_path = G1_XML_PATH
    
    if not xml_path.exists():
        print(f"❌ Error: Model file not found at {xml_path}")
        return
    
    print(f"Loading MuJoCo model from: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    
    print(f"Model loaded: {model.nq} DOF, {model.nu} actuators")
    
    # Find free joint
    free_joint_id = None
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            free_joint_id = i
            break
    
    if free_joint_id is None:
        print("⚠️  Warning: No free joint found")
    else:
        print(f"Found free joint (base) at joint index {free_joint_id}")
    
    # Disable gravity
    model.opt.gravity[:] = 0.0
    
    # Load CVAE model
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Error: Model file not found at {model_path}")
        return
    
    print(f"\nLoading CVAE model from: {model_path}")
    cvae_model = MotionCVAE(
        cond_dim=COND_DIM,
        future_dim=FUTURE_DIM,
        latent_dim=LATENT_DIM,
        hidden_dim=512,
        depth=4
    )
    
    checkpoint = torch.load(model_path, map_location=args.device)
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            cvae_model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            cvae_model.load_state_dict(checkpoint['state_dict'])
        else:
            cvae_model.load_state_dict(checkpoint)
    else:
        cvae_model.load_state_dict(checkpoint)
    
    cvae_model.eval()
    cvae_model.to(args.device)
    print("✅ CVAE model loaded")
    
    # Open joystick
    try:
        joystick = open_rumblepad()
        print("\n✅ Joystick connected!")
        print("\nControls:")
        print("  L STICK Y (vertical): Forward/Backward velocity (vx)")
        print("  L STICK X (horizontal): Yaw rotation (omega_z)")
        print("  R STICK X: Left/Right velocity (vy)")
        print("  Press Ctrl+C to quit\n")
    except Exception as e:
        print(f"❌ Failed to open joystick: {e}")
        return
    
    # Initialize robot pose
    if free_joint_id is not None:
        jnt_qposadr = model.jnt_qposadr[free_joint_id]
        data.qpos[jnt_qposadr + 3] = 1.0  # quaternion w
        data.qpos[jnt_qposadr + 2] = 0.165  # initial height
        mujoco.mj_forward(model, data)
    
    # Trajectory buffer for smooth playback
    trajectory_buffer = None
    trajectory_idx = 0
    last_valid_decoded = None
    
    # Main loop
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0, 0, 0.8]
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -20
        
        frame_count = 0
        last_inference_time = time.time()
        inference_interval = 0.1  # Generate new trajectory every 100ms
        
        try:
            while viewer.is_running():
                # Read joystick
                decoded = None
                if joystick:
                    report = joystick.read(64)
                    if report and len(report) >= 7:
                        decoded = decode_joystick(report)
                        if decoded is not None:
                            last_valid_decoded = decoded
                    
                    if decoded is None and last_valid_decoded is not None:
                        decoded = last_valid_decoded
                
                # Get joystick velocities
                vx, vy, omega_z = joystick_to_base_velocities(decoded)
                
                # Generate new trajectory if needed
                current_time = time.time()
                if (trajectory_buffer is None or 
                    trajectory_idx >= len(trajectory_buffer) or
                    current_time - last_inference_time >= inference_interval):
                    
                    # Get current robot state
                    q_current = data.qpos.copy()  # (nq,)
                    
                    # Prepare condition: [q_t (36), vx, vy, wz]
                    if len(q_current) >= D_CONF:
                        q_cond = q_current[:D_CONF]  # Use first D_CONF elements
                    else:
                        q_cond = np.pad(q_current, (0, max(0, D_CONF - len(q_current))))
                    
                    cond = np.concatenate([q_cond, [vx, vy, omega_z]])  # (39,)
                    cond_tensor = torch.FloatTensor(cond).unsqueeze(0).to(args.device)  # (1, 39)
                    
                    # Sample from latent space
                    with torch.no_grad():
                        z = torch.randn(1, LATENT_DIM).to(args.device)
                        future = cvae_model.decode(cond_tensor, z)  # (1, 30, 36)
                        future_np = future.cpu().numpy()[0]  # (30, 36)
                    
                    trajectory_buffer = future_np
                    trajectory_idx = 0
                    last_inference_time = current_time
                    
                    if frame_count % 60 == 0:  # Print every second
                        print(f"Generated trajectory | vx={vx:+.3f}, vy={vy:+.3f}, wz={omega_z:+.3f}")
                
                # Apply current frame from trajectory
                if trajectory_buffer is not None and trajectory_idx < len(trajectory_buffer):
                    q_target = trajectory_buffer[trajectory_idx]
                    
                    # Apply to robot (first D_CONF elements)
                    n_apply = min(len(q_target), model.nq)
                    data.qpos[:n_apply] = q_target[:n_apply]
                    
                    trajectory_idx += 1
                
                # Forward kinematics
                mujoco.mj_forward(model, data)
                
                # Update camera
                if free_joint_id is not None:
                    jnt_qposadr = model.jnt_qposadr[free_joint_id]
                    base_pos = data.qpos[jnt_qposadr:jnt_qposadr + 3]
                    viewer.cam.lookat[:] = base_pos.copy()
                    viewer.cam.lookat[2] += 0.8
                
                viewer.sync()
                frame_count += 1
                time.sleep(DT)
        
        except KeyboardInterrupt:
            print("\n\nSimulation stopped by user.")
        finally:
            if joystick:
                joystick.close()
            print("Joystick closed.")


if __name__ == "__main__":
    main()

