#!/usr/bin/env python3
"""
Joystick-controlled MuJoCo dummy simulation for Unitree G1.
This is a dummy simulation without dynamics - joints are directly controlled.
"""

import os
import time
import math
from pathlib import Path

import hid
import numpy as np
import mujoco
import mujoco.viewer

# ============================================================
# CONFIG
# ============================================================

FPS = 60.0
DT = 1.0 / FPS

# Joystick control parameters
AXIS_DEADZONE = 0.05  # Deadzone for joystick axes (reduced for better sensitivity)

# Base velocity control parameters
MAX_LIN_VEL = 1.0  # m/s - maximum linear velocity
MAX_ANG_VEL = 1.0  # rad/s - maximum angular velocity (yaw)

# Logitech Wireless Gamepad F710
# Note: F710 may appear as "Cordless RumblePad 2" in some systems
# F710 supports both XInput and DirectInput modes
# The HID report structure may vary based on the mode
VENDOR_ID = 0x046D
PRODUCT_ID = 0xC219  # F710 may use same PID as RumblePad 2, or will be auto-detected

# Path to MuJoCo model
MUJOCO_MENAGERIE_DIR = Path(__file__).resolve().parents[2] / "mujoco_menagerie"
G1_XML_PATH = MUJOCO_MENAGERIE_DIR / "unitree_g1" / "g1.xml"
G1_SCENE_XML_PATH = MUJOCO_MENAGERIE_DIR / "unitree_g1" / "scene.xml"


# ============================================================
# JOYSTICK HANDLING
# ============================================================

def find_logitech_gamepad():
    """Find Logitech gamepad (F710 or other) by enumerating HID devices."""
    for device_info in hid.enumerate():
        vid = device_info.get("vendor_id", 0)
        pid = device_info.get("product_id", 0)
        name = device_info.get("product_string", "")
        
        # Logitech vendor ID is 0x046D
        if vid == 0x046D:
            # Check if it's a gamepad (F710, RumblePad, etc.)
            if "gamepad" in name.lower() or "f710" in name.lower() or "rumblepad" in name.lower() or pid in [0xC219, 0xC21F, 0xC216]:
                return vid, pid, name
    
    return None, None, None

def open_rumblepad():
    """Open the Logitech gamepad (F710 or other)."""
    # Try to find the gamepad automatically
    vid, pid, name = find_logitech_gamepad()
    
    if vid is None:
        # Fallback to hardcoded values
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


def decode_joystick(report, debug=False):
    """Decode HID report from RumblePad."""
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

    # Debug: print raw report bytes
    if debug:
        print(f"Raw HID report (first 10 bytes): {list(report[:10])}")

    # Logitech F710 HID report structure (user confirmed):
    # byte_0 = buttons (low)
    # byte_1 = buttons (high) / axis0 (LEFT STICK X)
    # byte_2 = axis1 (LEFT STICK Y)
    # byte_3 = axis2 (RIGHT STICK X)
    # byte_4 = axis3 (RIGHT STICK Y)
    
    # Map bytes 1, 2, 3, 4 to axes 0, 1, 2, 3
    axis0 = _norm(report[1]) if len(report) > 1 else 0.0  # LEFT STICK X
    axis1 = _norm(report[2]) if len(report) > 2 else 0.0  # LEFT STICK Y
    axis2 = _norm(report[3]) if len(report) > 3 else 0.0  # RIGHT STICK X
    axis3 = _norm(report[4]) if len(report) > 4 else 0.0  # RIGHT STICK Y
    
    if debug:
        print(f"  LEFT STICK: X=byte_1={axis0:.3f} (raw={report[1] if len(report) > 1 else 'N/A'}), "
              f"Y=byte_2={axis1:.3f} (raw={report[2] if len(report) > 2 else 'N/A'})")
        print(f"  RIGHT STICK: X=byte_3={axis2:.3f} (raw={report[3] if len(report) > 3 else 'N/A'}), "
              f"Y=byte_4={axis3:.3f} (raw={report[4] if len(report) > 4 else 'N/A'})")
        print(f"  Raw bytes (0-9): {list(report[:10])}")

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
            axis0=axis0,   # L STICK AXIS 0 (X)
            axis1=axis1,   # L STICK AXIS 1 (Y)
            axis2=axis2,   # R STICK AXIS 2 (X)
            axis3=axis3,   # R STICK AXIS 3 (Y)
            lx=axis0, ly=axis1, rx=axis2, ry=axis3  # Keep old names for compatibility
        ),
        "buttons": buttons,
    }


# ============================================================
# BASE VELOCITY MAPPING
# ============================================================

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
# MAIN SIMULATION
# ============================================================

def main():
    # Load MuJoCo model - prefer scene.xml for grid background, fallback to g1.xml
    xml_path = G1_SCENE_XML_PATH if G1_SCENE_XML_PATH.exists() else G1_XML_PATH
    
    if not xml_path.exists():
        print(f"❌ Error: Model file not found at {xml_path}")
        print("Please ensure mujoco_menagerie is in the correct location.")
        return
    
    print(f"Loading MuJoCo model from: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    
    print(f"Model loaded: {model.nq} DOF, {model.nu} actuators")
    
    # Find the free joint (floating base) - it should be the first joint
    free_joint_id = None
    for i in range(model.njnt):
        jnt_type = model.jnt_type[i]
        if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
            free_joint_id = i
            break
    
    if free_joint_id is None:
        print("⚠️  Warning: No free joint found. Base velocity control may not work.")
    else:
        print(f"Found free joint (base) at joint index {free_joint_id}")
    
    # Disable gravity for dummy simulation
    model.opt.gravity[:] = 0.0
    
    # Set high position control gains for direct control (dummy mode)
    # The model already has position actuators with kp values
    
    # Initialize joint positions to zero (or neutral pose)
    data.qpos[:] = 0.0
    data.ctrl[:] = 0.0
    
    # Set initial base position so feet are on the ground
    if free_joint_id is not None:
        jnt_qposadr = model.jnt_qposadr[free_joint_id]
        # Set initial orientation: upright (w=1, x=0, y=0, z=0)
        data.qpos[jnt_qposadr + 3] = 1.0  # quaternion w
        data.qpos[jnt_qposadr + 4] = 0.0  # quaternion x
        data.qpos[jnt_qposadr + 5] = 0.0  # quaternion y
        data.qpos[jnt_qposadr + 6] = 0.0  # quaternion z
        
        # First, set pelvis to a reasonable height and compute forward kinematics
        # to find where the feet actually are
        data.qpos[jnt_qposadr + 0] = 0.0  # x
        data.qpos[jnt_qposadr + 1] = 0.0  # y
        data.qpos[jnt_qposadr + 2] = 0.8  # z (temporary, will adjust)
        
        # Forward kinematics to compute foot positions
        mujoco.mj_forward(model, data)
        
        # Find foot site positions
        left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
        
        if left_foot_id >= 0 and right_foot_id >= 0:
            # Get foot positions in world frame
            left_foot_pos = data.site_xpos[left_foot_id]
            right_foot_pos = data.site_xpos[right_foot_id]
            
            # Find the lowest foot point (foot geom is at -0.03 relative to ankle_roll)
            # So foot bottom is approximately 0.03m below the foot site
            min_foot_z = min(left_foot_pos[2], right_foot_pos[2]) - 0.03
            
            # Adjust pelvis z so that the lowest foot is at z=0 (ground level)
            current_pelvis_z = data.qpos[jnt_qposadr + 2]
            target_foot_z = 0.0
            adjustment = target_foot_z - min_foot_z
            new_pelvis_z = current_pelvis_z + adjustment
            
            data.qpos[jnt_qposadr + 2] = new_pelvis_z
            print(f"Adjusted pelvis z from {current_pelvis_z:.3f} to {new_pelvis_z:.3f} m")
            print(f"Foot positions: left={left_foot_pos[2]:.3f}, right={right_foot_pos[2]:.3f} m")
        else:
            # Fallback: use approximate calculation
            # pelvis(0.793) - hip(-0.1027) - knee(-0.17734) - ankle_pitch(-0.30001) - ankle_roll(-0.017558) - foot(-0.03)
            # = 0.793 - 0.1027 - 0.17734 - 0.30001 - 0.017558 - 0.03 = 0.1654
            data.qpos[jnt_qposadr + 2] = 0.165
            print("⚠️  Could not find foot sites, using approximate height")
    
    # Forward kinematics with final position
    mujoco.mj_forward(model, data)
    
    # Open joystick
    try:
        joystick = open_rumblepad()
        print("\n✅ Joystick connected!")
        print("\nBase Velocity Controls:")
        print("  L STICK AXIS 1 (Y, vertical): Forward/Backward velocity (vx)")
        print("  L STICK AXIS 0 (X, horizontal): Yaw rotation (omega_z)")
        print("  R STICK AXIS 2: Left/Right velocity (vy)")
        print("  Press Ctrl+C to quit")
        print("\nVelocity output at 10 Hz:")
        print("(If joystick doesn't respond, check if axis values change when moving sticks)\n")
    except Exception as e:
        print(f"⚠️  Warning: Could not open joystick: {e}")
        print("Continuing without joystick (robot will be stationary)")
        joystick = None
    
    # Run simulation with viewer
    print("Starting MuJoCo viewer...")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set initial camera position
        viewer.cam.lookat[:] = [0, 0, 0.8]  # Look at robot center
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -20
        
        frame_count = 0
        start_time = time.time()
        last_print_time = time.time()
        print_interval = 1.0 / 10.0  # 10 Hz = 0.1 seconds
        
        # Debug mode: set to True to see raw HID data
        debug_joystick = False
        debug_count = 0
        
        # Store last valid joystick state to avoid "no joystick" messages
        # when MuJoCo loop runs faster than joystick report rate (MuJoCo ~60Hz, joystick ~10-20Hz)
        last_valid_decoded = None
        
        try:
            while viewer.is_running():
                # Read joystick input (non-blocking)
                # Only use input when new data is received, otherwise use None (zero velocity)
                decoded = None
                if joystick:
                    report = joystick.read(64)
                    if report and len(report) >= 7:  # Only use if we got valid new data
                        # Debug: print first few reports to diagnose
                        if debug_joystick and debug_count < 10:
                            print(f"\n=== Debug Report #{debug_count} ===")
                            decoded = decode_joystick(report, debug=True)
                            debug_count += 1
                        else:
                            decoded = decode_joystick(report, debug=False)
                        
                        # Update last valid state when we get new data
                        if decoded is not None:
                            last_valid_decoded = decoded
                    
                    # Use last valid state if no new report (MuJoCo loop is faster than joystick)
                    # But only if we've received at least one valid report before
                    if decoded is None and last_valid_decoded is not None:
                        decoded = last_valid_decoded
                
                # Get base velocities
                vx, vy, omega_z = joystick_to_base_velocities(decoded)
                
                # Print velocities at 10 Hz (new line each time for better visibility)
                current_time = time.time()
                if current_time - last_print_time >= print_interval:
                    # Also print raw axis values for debugging
                    if decoded and "axes" in decoded:
                        ax = decoded["axes"]
                        axis0_val = ax.get("axis0", ax.get("lx", 0.0))
                        axis1_val = ax.get("axis1", ax.get("ly", 0.0))
                        axis2_val = ax.get("axis2", ax.get("rx", 0.0))
                        axis3_val = ax.get("axis3", ax.get("ry", 0.0))
                        print(f"vx: {vx:+.3f} m/s, vy: {vy:+.3f} m/s, wz: {omega_z:+.3f} rad/s | "
                              f"axis0: {axis0_val:+.3f}, axis1: {axis1_val:+.3f}, axis2: {axis2_val:+.3f}, axis3: {axis3_val:+.3f}")
                    else:
                        # Only show "no joystick" if we've never received any data
                        if last_valid_decoded is None:
                            print(f"vx: {vx:+.3f} m/s, vy: {vy:+.3f} m/s, wz: {omega_z:+.3f} rad/s (no joystick)")
                        else:
                            # We have last valid data, just no new report this frame
                            ax = last_valid_decoded["axes"]
                            axis0_val = ax.get("axis0", ax.get("lx", 0.0))
                            axis1_val = ax.get("axis1", ax.get("ly", 0.0))
                            axis2_val = ax.get("axis2", ax.get("rx", 0.0))
                            axis3_val = ax.get("axis3", ax.get("ry", 0.0))
                            print(f"vx: {vx:+.3f} m/s, vy: {vy:+.3f} m/s, wz: {omega_z:+.3f} rad/s | "
                                  f"axis0: {axis0_val:+.3f}, axis1: {axis1_val:+.3f}, axis2: {axis2_val:+.3f}, axis3: {axis3_val:+.3f}")
                    last_print_time = current_time
                
                # Update base position and orientation
                if free_joint_id is not None:
                    # Free joint: qpos[0:3] = position, qpos[3:7] = quaternion (w, x, y, z)
                    jnt_qposadr = model.jnt_qposadr[free_joint_id]
                    
                    # Get current position and orientation
                    pos = data.qpos[jnt_qposadr:jnt_qposadr + 3].copy()
                    quat = data.qpos[jnt_qposadr + 3:jnt_qposadr + 7].copy()
                    
                    # Normalize quaternion
                    quat_norm = np.linalg.norm(quat)
                    if quat_norm > 1e-6:
                        quat = quat / quat_norm
                    
                    # Extract yaw angle from quaternion (for rotating velocity vector)
                    # Quaternion to yaw: atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
                    w, x, y, z = quat
                    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                    
                    # Convert body-frame velocities to world frame
                    # vx is forward in body frame, vy is left/right in body frame
                    cos_yaw = math.cos(yaw)
                    sin_yaw = math.sin(yaw)
                    vx_world = vx * cos_yaw - vy * sin_yaw  # Forward component
                    vy_world = vx * sin_yaw + vy * cos_yaw  # Left/right component
                    
                    # Update position in world frame
                    pos[0] += vx_world * DT  # Forward/backward in world
                    pos[1] += vy_world * DT  # Left/right in world
                    # z position stays the same (or could add control later)
                    
                    # Update yaw rotation
                    if abs(omega_z) > 1e-6:
                        # Create rotation quaternion for yaw
                        yaw_angle = omega_z * DT
                        cos_half = math.cos(yaw_angle / 2.0)
                        sin_half = math.sin(yaw_angle / 2.0)
                        yaw_quat = np.array([cos_half, 0.0, 0.0, sin_half])  # Rotation around z-axis
                        
                        # Multiply quaternions: quat_new = quat * yaw_quat
                        w1, x1, y1, z1 = quat
                        w2, x2, y2, z2 = yaw_quat
                        quat = np.array([
                            w1*w2 - x1*x2 - y1*y2 - z1*z2,
                            w1*x2 + x1*w2 + y1*z2 - z1*y2,
                            w1*y2 - x1*z2 + y1*w2 + z1*x2,
                            w1*z2 + x1*y2 - y1*x2 + z1*w2
                        ])
                        
                        # Normalize again
                        quat = quat / np.linalg.norm(quat)
                    
                    # Update qpos
                    data.qpos[jnt_qposadr:jnt_qposadr + 3] = pos
                    data.qpos[jnt_qposadr + 3:jnt_qposadr + 7] = quat
                
                # Forward kinematics (no physics step for dummy simulation)
                mujoco.mj_forward(model, data)
                
                # Update camera to follow robot base
                if free_joint_id is not None:
                    jnt_qposadr = model.jnt_qposadr[free_joint_id]
                    base_pos = data.qpos[jnt_qposadr:jnt_qposadr + 3]
                    viewer.cam.lookat[:] = base_pos.copy()
                    viewer.cam.lookat[2] += 0.8  # Look slightly above base
                
                # Sync viewer
                viewer.sync()
                
                # Rate limiting
                frame_count += 1
                elapsed = time.time() - start_time
                if elapsed > 0:
                    actual_fps = frame_count / elapsed
                    if frame_count % 60 == 0:
                        print(f"FPS: {actual_fps:.1f}", end='\r')
                
                time.sleep(DT)
        
        except KeyboardInterrupt:
            print("\n\nSimulation stopped by user.")
        finally:
            if joystick:
                joystick.close()
            print("Joystick closed.")


if __name__ == "__main__":
    main()

