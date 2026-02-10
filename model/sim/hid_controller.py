#!/usr/bin/env python3
"""
HID controller utilities for Windows + conda.
Works with: hid==1.0.8 (pyhidapi) which exposes hid.enumerate() + hid.Device(...)

Outputs sticks (lx,ly,rx,ry) in ~[-1,1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import hid

LOGITECH_VID = 0x046D
SONY_VID = 0x054C


def _norm_u8(v: int) -> float:
    return (int(v) - 128) / 128.0


def _apply_deadzone(x: float, deadzone: float) -> float:
    return 0.0 if abs(x) < deadzone else x


@dataclass
class ControllerAxes:
    lx: float
    ly: float
    rx: float
    ry: float


@dataclass
class ControllerState:
    axes: ControllerAxes
    raw: bytes


def choose_controller_profile(device_info: Dict[str, Any]) -> str:
    vid = device_info.get("vendor_id", 0)
    name = (device_info.get("product_string") or "").lower()
    if vid == SONY_VID or "wireless controller" in name or "dualshock" in name:
        return "ps4"
    if vid == LOGITECH_VID:
        return "logitech"
    return "generic"


def find_any_gamepad() -> Optional[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for d in hid.enumerate():
        vid = d.get("vendor_id", 0)
        name = (d.get("product_string") or "")
        usage_page = d.get("usage_page", None)
        usage = d.get("usage", None)

        looks_like_controller = (
            ("controller" in name.lower())
            or ("gamepad" in name.lower())
            or ("joystick" in name.lower())
            or ("f710" in name.lower())
            or ("rumblepad" in name.lower())
        )
        is_generic_desktop = (usage_page == 0x01 and usage in (0x04, 0x05))  # joystick/gamepad

        if looks_like_controller or is_generic_desktop or vid in (SONY_VID, LOGITECH_VID):
            candidates.append(d)

    if not candidates:
        return None

    def score(d: Dict[str, Any]) -> int:
        vid = d.get("vendor_id", 0)
        nm = (d.get("product_string") or "").lower()
        s = 0
        if vid == SONY_VID:
            s += 100
        if vid == LOGITECH_VID:
            s += 90
        if "wireless controller" in nm:
            s += 30
        if "dualshock" in nm:
            s += 30
        if "f710" in nm:
            s += 20
        if "gamepad" in nm:
            s += 10
        if "controller" in nm:
            s += 10
        return s

    candidates.sort(key=score, reverse=True)
    return candidates[0]


def open_hid_device(device_info: Dict[str, Any]):
    """
    pyhidapi open:
      hid.Device(path=...) preferred
    """
    path = device_info.get("path", None)
    vid = device_info.get("vendor_id", 0)
    pid = device_info.get("product_id", 0)

    if hasattr(hid, "Device"):
        if path is not None:
            return hid.Device(path=path)
        return hid.Device(vid=vid, pid=pid)

    raise RuntimeError("Expected pyhidapi (hid.Device missing).")


def open_controller() -> Tuple[object, str, Dict[str, Any]]:
    info = find_any_gamepad()
    if info is None:
        raise RuntimeError("No HID controller detected via hid.enumerate().")

    profile = choose_controller_profile(info)
    name = info.get("product_string", "")
    vid = info.get("vendor_id", 0)
    pid = info.get("product_id", 0)
    print(f"✅ Selected HID device: {name} (VID={vid:04X}, PID={pid:04X}), profile='{profile}'")

    dev = open_hid_device(info)
    return dev, profile, info


def decode_report(report: bytes, profile: str, deadzone: float, debug: bool = False) -> Optional[ControllerState]:
    """
    Decode sticks. DS4 USB reports vary; we try both offsets [1..4] and [0..3].
    """
    if not report or len(report) < 5:
        return None

    def cand_shift0() -> Optional[ControllerAxes]:
        if len(report) < 5:
            return None
        return ControllerAxes(
            lx=_norm_u8(report[1]),
            ly=_norm_u8(report[2]),
            rx=_norm_u8(report[3]),
            ry=_norm_u8(report[4]),
        )

    def cand_shift_minus1() -> Optional[ControllerAxes]:
        if len(report) < 4:
            return None
        return ControllerAxes(
            lx=_norm_u8(report[0]),
            ly=_norm_u8(report[1]),
            rx=_norm_u8(report[2]),
            ry=_norm_u8(report[3]),
        )

    def score(ax: Optional[ControllerAxes]) -> float:
        if ax is None:
            return -1.0
        return abs(ax.lx) + abs(ax.ly) + abs(ax.rx) + abs(ax.ry)

    if profile == "logitech":
        ax = cand_shift0()
    elif profile == "ps4":
        a0 = cand_shift0()
        a1 = cand_shift_minus1()
        # If report[0] looks like a small report_id, prefer shift0
        if 1 <= report[0] <= 8 and a0 is not None:
            ax = a0
        else:
            ax = a0 if score(a0) >= score(a1) else a1
    else:
        a0 = cand_shift0()
        a1 = cand_shift_minus1()
        ax = a0 if score(a0) >= score(a1) else a1

    if ax is None:
        return None

    ax = ControllerAxes(
        lx=_apply_deadzone(ax.lx, deadzone),
        ly=_apply_deadzone(ax.ly, deadzone),
        rx=_apply_deadzone(ax.rx, deadzone),
        ry=_apply_deadzone(ax.ry, deadzone),
    )

    if debug:
        print(f"Raw HID (0..9): {list(report[:10])} | lx={ax.lx:+.3f} ly={ax.ly:+.3f} rx={ax.rx:+.3f} ry={ax.ry:+.3f}")

    return ControllerState(axes=ax, raw=report)


def joystick_to_base_velocities(state: Optional[ControllerState], max_lin_vel: float, max_ang_vel: float):
    """
    Same mapping you’ve been using:
      vx = -ly * max_lin_vel
      vy = -rx * max_lin_vel
      wz = -lx * max_ang_vel
    """
    if state is None:
        return 0.0, 0.0, 0.0

    lx, ly, rx = state.axes.lx, state.axes.ly, state.axes.rx
    vx = -ly * max_lin_vel
    vy = -rx * max_lin_vel
    wz = -lx * max_ang_vel
    return vx, vy, wz
