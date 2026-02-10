#!/usr/bin/env python3
"""
Joystick-controlled MuJoCo dummy simulation for Unitree G1.

Minimal-change version that works with:
- Logitech F710 (Logitech vendor 0x046D)
- PS4 DualShock 4 controller over USB (Sony vendor 0x054C)

Notes:
- This is still a "dummy" sim: we directly update base qpos (no dynamics), and call mj_forward.
- PS4 report formats can differ (USB vs Bluetooth, DS4Windows/Steam remapping). This code handles
  the common USB case and includes a small auto-detect fallback.
"""

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

AXIS_DEADZONE = 0.05

MAX_LIN_VEL = 1.0   # m/s
MAX_ANG_VEL = 1.0   # rad/s

# Vendor IDs
LOGITECH_VID = 0x046D
SONY_VID     = 0x054C  # DualShock 4 (PS4) is typically Sony VID

# Path to MuJoCo model (repo-root mujoco_menagerie)
MUJOCO_MENAGERIE_DIR = Path(__file__).resolve().parents[2] / "mujoco_menagerie"
if not MUJOCO_MENAGERIE_DIR.exists():
    MUJOCO_MENAGERIE_DIR = Path(__file__).resolve().parents[3] / "mujoco_menagerie"

G1_XML_PATH = MUJOCO_MENAGERIE_DIR / "unitree_g1" / "g1.xml"
G1_SCENE_XML_PATH = MUJOCO_MENAGERIE_DIR / "unitree_g1" / "scene.xml"

# ============================================================
# JOYSTICK HANDLING
# ============================================================

def open_hid_device(device_info):
    """
    Open a HID device using the 'hid' package we have (pyhidapi: hid==1.0.8).
    device_info comes from hid.enumerate() and includes vendor_id/product_id/path.
    """
    # pyhidapi provides hid.Device(...)
    path = device_info.get("path", None)
    vid  = device_info.get("vendor_id", 0)
    pid  = device_info.get("product_id", 0)

    if hasattr(hid, "Device"):
        if path is not None:
            return hid.Device(path=path)
        return hid.Device(vid=vid, pid=pid)

    # Fallback: if you *do* have a different hid binding installed
    if hasattr(hid, "device"):
        dev = hid.device()
        if path is not None and hasattr(dev, "open_path"):
            dev.open_path(path)
        else:
            dev.open(vid, pid)
        dev.set_nonblocking(True)
        return dev

    raise RuntimeError("Your 'hid' module doesn't expose hid.Device or hid.device().")

def _norm_u8(v: int) -> float:
    """Normalize unsigned byte 0..255 to approx -1..+1 centered at 128."""
    return (int(v) - 128) / 128.0

def apply_deadzone(val: float, deadzone: float) -> float:
    return 0.0 if abs(val) < deadzone else val

def choose_controller_profile(device_info: dict) -> str:
    """
    Decide how to decode reports. We keep it simple:
    - Logitech F710: "logitech"
    - Sony DualShock 4: "ps4"
    - otherwise: "generic"
    """
    vid = device_info.get("vendor_id", 0)
    name = (device_info.get("product_string") or "").lower()
    if vid == LOGITECH_VID:
        return "logitech"
    if vid == SONY_VID or "wireless controller" in name or "dualshock" in name:
        return "ps4"
    return "generic"

def find_any_gamepad():
    """
    Find a reasonable HID gamepad device.

    Instead of hard-coding Logitech PID, we:
    - scan devices
    - prefer Sony/Logitech if present
    - fall back to anything that looks like a controller by name/usage_page
    """
    candidates = []
    for d in hid.enumerate():
        vid = d.get("vendor_id", 0)
        name = (d.get("product_string") or "")
        usage_page = d.get("usage_page", None)
        usage = d.get("usage", None)

        # Heuristics: likely controller name OR common HID gamepad usages (0x01 generic desktop, usage 0x05 gamepad / 0x04 joystick)
        looks_like_controller = (
            ("controller" in name.lower())
            or ("gamepad" in name.lower())
            or ("joystick" in name.lower())
            or ("f710" in name.lower())
            or ("rumblepad" in name.lower())
        )

        is_generic_desktop = (usage_page == 0x01 and usage in (0x04, 0x05))

        if looks_like_controller or is_generic_desktop or vid in (LOGITECH_VID, SONY_VID):
            candidates.append(d)

    if not candidates:
        return None

    # Prefer Sony/Logitech first
    def score(d):
        vid = d.get("vendor_id", 0)
        name = (d.get("product_string") or "").lower()
        s = 0
        if vid == SONY_VID: s += 100
        if vid == LOGITECH_VID: s += 90
        if "wireless controller" in name: s += 20
        if "dualshock" in name: s += 20
        if "f710" in name: s += 15
        if "gamepad" in name: s += 10
        if "controller" in name: s += 10
        return s

    candidates.sort(key=score, reverse=True)
    return candidates[0]

def open_gamepad():
    d = find_any_gamepad()
    if d is None:
        raise RuntimeError("No HID gamepad/controller detected by hid.enumerate().")

    vid = d.get("vendor_id", 0)
    pid = d.get("product_id", 0)
    name = d.get("product_string", "")
    path = d.get("path", None)

    profile = choose_controller_profile(d)

    print(f"✅ Selected HID device: {name} (VID={vid:04X}, PID={pid:04X}), profile='{profile}'")

    dev = open_hid_device(d)
    # pyhidapi is blocking by default; set nonblocking if supported
    if hasattr(dev, "set_nonblocking"):
        dev.set_nonblocking(True)
    return dev, profile

def decode_axes(report: bytes, profile: str, debug: bool = False):
    """
    Return normalized axes (lx, ly, rx, ry) in [-1,1] approximately, plus buttons list (optional).
    Minimal: we focus on sticks only.

    Profiles:
    - logitech: original code assumed:
        report[1]=LX, report[2]=LY, report[3]=RX, report[4]=RY
      BUT it also treated report[0]/[1] as buttons in some modes. Here we just read axes.
    - ps4: common DS4 USB report:
        report[0]=report_id (often 0x01), report[1]=LX, report[2]=LY, report[3]=RX, report[4]=RY
      Some setups may omit report_id; we handle both by auto-shifting if needed.
    - generic: try both shifts and pick the one with "more movement" (non-zero variance).
    """
    if not report or len(report) < 6:
        return None

    # Helper to extract with a given shift
    def extract_with_shift(shift: int):
        # Need indices shift+1..shift+4
        if len(report) <= shift + 4:
            return None
        lx = _norm_u8(report[shift + 1])
        ly = _norm_u8(report[shift + 2])
        rx = _norm_u8(report[shift + 3])
        ry = _norm_u8(report[shift + 4])
        return lx, ly, rx, ry

    if profile == "ps4":
        # DS4 USB often has report_id at [0] == 0x01
        # If [0] looks like a small report_id (1..5), use shift=0 (axes start at 1).
        # If not, try shift=-1 style (axes start at 0) by using shift=-1 equivalent: shift=-1 -> indices 0..3.
        # We'll implement both and choose the sensible one.
        cand0 = extract_with_shift(0)  # axes at 1..4
        cand1 = None
        if len(report) >= 4:
            # axes at 0..3
            lx = _norm_u8(report[0])
            ly = _norm_u8(report[1])
            rx = _norm_u8(report[2])
            ry = _norm_u8(report[3])
            cand1 = (lx, ly, rx, ry)

        # Choose candidate: if report[0] is a small report id, prefer cand0.
        if 1 <= report[0] <= 8 and cand0 is not None:
            lx, ly, rx, ry = cand0
        else:
            # otherwise prefer the one that isn't all zeros / deadzone
            def score(c):
                if c is None: return -1
                return sum(abs(v) for v in c)
            best = cand0 if score(cand0) >= score(cand1) else cand1
            if best is None:
                return None
            lx, ly, rx, ry = best

    elif profile == "logitech":
        # Many Logitech modes: sticks often land at 1..4 as well.
        cand = extract_with_shift(0)
        if cand is None:
            return None
        lx, ly, rx, ry = cand

    else:
        # generic: try axes at 1..4 and 0..3, pick the one with bigger magnitude
        cand0 = extract_with_shift(0)
        cand1 = None
        if len(report) >= 4:
            cand1 = (_norm_u8(report[0]), _norm_u8(report[1]), _norm_u8(report[2]), _norm_u8(report[3]))

        def score(c):
            if c is None: return -1
            return sum(abs(v) for v in c)

        best = cand0 if score(cand0) >= score(cand1) else cand1
        if best is None:
            return None
        lx, ly, rx, ry = best

    # Deadzone
    lx = apply_deadzone(lx, AXIS_DEADZONE)
    ly = apply_deadzone(ly, AXIS_DEADZONE)
    rx = apply_deadzone(rx, AXIS_DEADZONE)
    ry = apply_deadzone(ry, AXIS_DEADZONE)

    if debug:
        print(f"Raw HID (0..9): {list(report[:10])} | lx={lx:+.3f} ly={ly:+.3f} rx={rx:+.3f} ry={ry:+.3f}")

    return {"axes": {"lx": lx, "ly": ly, "rx": rx, "ry": ry}, "buttons": []}

# ============================================================
# BASE VELOCITY MAPPING
# ============================================================

def joystick_to_base_velocities(decoded):
    """
    Returns (vx, vy, omega_z)
    - vx: forward/backward from LY
    - vy: left/right from RX
    - omega_z: yaw from LX
    """
    if decoded is None:
        return 0.0, 0.0, 0.0

    ax = decoded["axes"]
    lx, ly, rx = ax["lx"], ax["ly"], ax["rx"]

    vx = -ly * MAX_LIN_VEL        # up on stick => forward
    vy = -rx * MAX_LIN_VEL        # right stick right => negative vy (kept from your original)
    omega_z = -lx * MAX_ANG_VEL   # left stick right => negative yaw (kept from your original)

    return vx, vy, omega_z

# ============================================================
# MAIN SIMULATION
# ============================================================

def main():
    xml_path = G1_SCENE_XML_PATH if G1_SCENE_XML_PATH.exists() else G1_XML_PATH
    if not xml_path.exists():
        print(f"❌ Error: Model file not found at {xml_path}")
        print(f"Checked MUJOCO_MENAGERIE_DIR={MUJOCO_MENAGERIE_DIR}")
        return

    print(f"Loading MuJoCo model from: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    print(f"Model loaded: nq={model.nq}, nu={model.nu}, njnt={model.njnt}")

    # Find free joint (floating base)
    free_joint_id = None
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            free_joint_id = i
            break
    if free_joint_id is None:
        print("⚠️  Warning: No free joint found. Base velocity control may not work.")
    else:
        print(f"Found free joint (base) at joint index {free_joint_id}")

    # Dummy mode: no gravity
    model.opt.gravity[:] = 0.0

    # Init state
    data.qpos[:] = 0.0
    if model.nu > 0:
        data.ctrl[:] = 0.0

    # Put base upright + feet on ground (uses foot sites if present)
    if free_joint_id is not None:
        adr = model.jnt_qposadr[free_joint_id]
        data.qpos[adr + 3] = 1.0  # qw
        data.qpos[adr + 4] = 0.0
        data.qpos[adr + 5] = 0.0
        data.qpos[adr + 6] = 0.0

        data.qpos[adr + 0] = 0.0
        data.qpos[adr + 1] = 0.0
        data.qpos[adr + 2] = 0.8  # temp

        mujoco.mj_forward(model, data)

        left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

        if left_foot_id >= 0 and right_foot_id >= 0:
            lf = data.site_xpos[left_foot_id]
            rf = data.site_xpos[right_foot_id]
            min_foot_z = min(lf[2], rf[2]) - 0.03  # approx sole below site
            adjustment = 0.0 - min_foot_z
            data.qpos[adr + 2] = data.qpos[adr + 2] + adjustment
            print(f"Adjusted pelvis z to {data.qpos[adr+2]:.3f} (feet ~0)")
        else:
            data.qpos[adr + 2] = 0.165
            print("⚠️  Could not find foot sites, using approximate height")

    mujoco.mj_forward(model, data)

    # Open controller
    joystick = None
    profile = "generic"
    try:
        joystick, profile = open_gamepad()
        print("\n✅ Controller connected!")
        print("Controls:")
        print("  Left stick Y: forward/back (vx)")
        print("  Left stick X: yaw (wz)")
        print("  Right stick X: left/right (vy)")
        print("  Ctrl+C to quit\n")
    except Exception as e:
        print(f"❌ Could not open controller via hid: {e}")
        print("   (If you *must* use the PS4 controller, ensure it's connected via USB and visible in hid.enumerate())")
        return

    print("Starting MuJoCo viewer...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0, 0, 0.8]
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -20

        last_print_time = time.time()
        print_interval = 0.1  # 10 Hz

        debug_joystick = False  # set True to see raw bytes + decoded axes

        last_valid = None

        try:
            while viewer.is_running():
                decoded = None
                try:
                    report = joystick.read(64)
                except TypeError:
                # pyhidapi signature can be read(size, timeout_ms)
                    report = joystick.read(64, 0)
                if report:
                    decoded = decode_axes(report, profile=profile, debug=debug_joystick)
                    if decoded is not None:
                        last_valid = decoded

                if decoded is None and last_valid is not None:
                    decoded = last_valid

                vx, vy, omega_z = joystick_to_base_velocities(decoded)

                now = time.time()
                if now - last_print_time >= print_interval:
                    if decoded is not None:
                        ax = decoded["axes"]
                        print(
                            f"vx {vx:+.3f}  vy {vy:+.3f}  wz {omega_z:+.3f} | "
                            f"lx {ax['lx']:+.3f}  ly {ax['ly']:+.3f}  rx {ax['rx']:+.3f}  ry {ax['ry']:+.3f}"
                        )
                    else:
                        print(f"vx {vx:+.3f}  vy {vy:+.3f}  wz {omega_z:+.3f} (no report yet)")
                    last_print_time = now

                # Update base pose
                if free_joint_id is not None:
                    adr = model.jnt_qposadr[free_joint_id]
                    pos = data.qpos[adr:adr + 3].copy()
                    quat = data.qpos[adr + 3:adr + 7].copy()

                    # Normalize quat
                    n = np.linalg.norm(quat)
                    if n > 1e-6:
                        quat = quat / n

                    w, x, y, z = quat
                    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

                    cy = math.cos(yaw)
                    sy = math.sin(yaw)
                    vx_world = vx * cy - vy * sy
                    vy_world = vx * sy + vy * cy

                    pos[0] += vx_world * DT
                    pos[1] += vy_world * DT

                    if abs(omega_z) > 1e-6:
                        yaw_angle = omega_z * DT
                        cos_half = math.cos(yaw_angle / 2.0)
                        sin_half = math.sin(yaw_angle / 2.0)
                        yaw_quat = np.array([cos_half, 0.0, 0.0, sin_half])  # about z

                        w1, x1, y1, z1 = quat
                        w2, x2, y2, z2 = yaw_quat
                        quat = np.array([
                            w1*w2 - x1*x2 - y1*y2 - z1*z2,
                            w1*x2 + x1*w2 + y1*z2 - z1*y2,
                            w1*y2 - x1*z2 + y1*w2 + z1*x2,
                            w1*z2 + x1*y2 - y1*x2 + z1*w2
                        ])
                        quat = quat / np.linalg.norm(quat)

                    data.qpos[adr:adr + 3] = pos
                    data.qpos[adr + 3:adr + 7] = quat

                mujoco.mj_forward(model, data)

                # Camera follow
                if free_joint_id is not None:
                    adr = model.jnt_qposadr[free_joint_id]
                    base_pos = data.qpos[adr:adr + 3]
                    viewer.cam.lookat[:] = base_pos.copy()
                    viewer.cam.lookat[2] += 0.8

                viewer.sync()
                time.sleep(DT)

        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            try:
                joystick.close()
            except Exception:
                pass
            print("Controller closed.")

if __name__ == "__main__":
    main()