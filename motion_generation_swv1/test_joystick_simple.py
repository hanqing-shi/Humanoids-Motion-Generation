#!/usr/bin/env python3
"""
Simple joystick test script - tests HID joystick input without MuJoCo.
Records all axis values and button presses to a log file.
"""

import hid
import time
import csv
from datetime import datetime
from pathlib import Path

# Logitech Wireless Gamepad F710
# Note: F710 may appear as "Cordless RumblePad 2" in some systems
VENDOR_ID = 0x046D
PRODUCT_ID = 0xC219  # F710 may use same PID, or will be auto-detected

# Output file
OUTPUT_DIR = Path(__file__).parent
LOG_FILE = OUTPUT_DIR / f"joystick_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


def _norm(v: int) -> float:
    """Normalize 0~255 -> -1.0~+1.0."""
    return (v - 128) / 128.0


def decode_joystick(report):
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

    # Try all possible byte positions for analog sticks
    # Return all possible mappings
    result = {
        "buttons": buttons,
        "raw_bytes": list(report[:16]) if len(report) >= 16 else list(report),
        "mappings": {}
    }
    
    # Mapping 1: report[2-5]
    if len(report) >= 6:
        result["mappings"]["map1_bytes_2_5"] = {
            "axis0": _norm(report[2]),
            "axis1": _norm(report[3]),
            "axis2": _norm(report[4]),
            "axis3": _norm(report[5]),
        }
    
    # Mapping 2: report[4-7]
    if len(report) >= 8:
        result["mappings"]["map2_bytes_4_7"] = {
            "axis0": _norm(report[4]),
            "axis1": _norm(report[5]),
            "axis2": _norm(report[6]),
            "axis3": _norm(report[7]),
        }
    
    # Mapping 3: report[6-9]
    if len(report) >= 10:
        result["mappings"]["map3_bytes_6_9"] = {
            "axis0": _norm(report[6]),
            "axis1": _norm(report[7]),
            "axis2": _norm(report[8]),
            "axis3": _norm(report[9]),
        }
    
    # Also try individual bytes
    result["all_bytes_normalized"] = {}
    for i in range(min(16, len(report))):
        result["all_bytes_normalized"][f"byte_{i}"] = _norm(report[i])
    
    return result


def find_logitech_gamepad():
    """Find Logitech gamepad (F710 or other) by enumerating HID devices."""
    print("\nSearching for Logitech gamepads...")
    for device_info in hid.enumerate():
        vid = device_info.get("vendor_id", 0)
        pid = device_info.get("product_id", 0)
        name = device_info.get("product_string", "")
        mfg = device_info.get("manufacturer_string", "")
        
        # Logitech vendor ID is 0x046D
        if vid == 0x046D:
            print(f"  Found: {name} (VID={vid:04X}, PID={pid:04X}, Manufacturer={mfg})")
            # Check if it's a gamepad
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
        print(f"✅ Using: {name} (VID={vid:04X}, PID={pid:04X})")
    
    dev = hid.device()
    dev.open(vid, pid)
    dev.set_nonblocking(True)
    return dev


def main():
    print("=" * 60)
    print("Joystick Test Script")
    print("=" * 60)
    print(f"Looking for Logitech RumblePad 2 (VID: {VENDOR_ID:04X}, PID: {PRODUCT_ID:04X})")
    print()
    
    # Open joystick
    try:
        joystick = open_rumblepad()
        print("✅ Joystick connected!")
    except Exception as e:
        print(f"❌ Failed to open joystick: {e}")
        print("\nTroubleshooting:")
        print("1. Check if joystick is connected")
        print("2. Check USB receiver")
        print("3. Try: python test_joy.py (to list all HID devices)")
        return
    
    print(f"\n📝 Logging to: {LOG_FILE}")
    print("\nControls:")
    print("  - Move joystick sticks to see axis values")
    print("  - Press buttons to see button numbers")
    print("  - Press Ctrl+C to stop and save log")
    print("\n" + "=" * 60)
    print("Starting capture... (Press Ctrl+C to stop)\n")
    
    # Open CSV file for logging
    csv_file = open(LOG_FILE, 'w', newline='')
    csv_writer = None
    
    sample_count = 0
    last_report_time = time.time()
    
    try:
        while True:
            report = joystick.read(64)
            
            if report and len(report) >= 7:
                decoded = decode_joystick(report)
                
                if decoded:
                    current_time = time.time()
                    elapsed = current_time - last_report_time
                    last_report_time = current_time
                    
                    sample_count += 1
                    
                    # Print to console
                    print(f"\n[Sample #{sample_count}, Δt={elapsed*1000:.1f}ms]")
                    print(f"  Buttons: {decoded['buttons'] if decoded['buttons'] else 'None'}")
                    
                    # Print all mappings
                    for map_name, axes in decoded['mappings'].items():
                        print(f"  {map_name}:")
                        print(f"    axis0: {axes['axis0']:+.3f}, axis1: {axes['axis1']:+.3f}, "
                              f"axis2: {axes['axis2']:+.3f}, axis3: {axes['axis3']:+.3f}")
                    
                    # Print raw bytes with highlighting for changing values
                    raw_bytes = decoded['raw_bytes'][:16]
                    print(f"  Raw bytes (0-15): {raw_bytes}")
                    
                    # Show which bytes are changing (compared to previous sample)
                    if sample_count > 1:
                        # This will be shown in next iteration, but we can track it
                        pass
                    
                    # Print normalized byte values for easier debugging
                    print(f"  Normalized bytes:")
                    for i, byte_val in enumerate(raw_bytes[:10]):
                        norm_val = _norm(byte_val)
                        if abs(norm_val) > 0.01:  # Only show non-zero values
                            print(f"    byte_{i}: {byte_val:3d} -> {norm_val:+.3f}")
                    
                    # Initialize CSV writer on first sample
                    if csv_writer is None:
                        # Create header
                        fieldnames = ['sample', 'timestamp', 'elapsed_ms', 'buttons']
                        for map_name in decoded['mappings'].keys():
                            fieldnames.extend([f"{map_name}_axis0", f"{map_name}_axis1", 
                                             f"{map_name}_axis2", f"{map_name}_axis3"])
                        fieldnames.extend([f"byte_{i}" for i in range(min(16, len(decoded['raw_bytes'])))])
                        
                        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                        csv_writer.writeheader()
                    
                    # Write to CSV
                    row = {
                        'sample': sample_count,
                        'timestamp': current_time,
                        'elapsed_ms': elapsed * 1000,
                        'buttons': ','.join(map(str, decoded['buttons'])) if decoded['buttons'] else ''
                    }
                    
                    for map_name, axes in decoded['mappings'].items():
                        row[f"{map_name}_axis0"] = axes['axis0']
                        row[f"{map_name}_axis1"] = axes['axis1']
                        row[f"{map_name}_axis2"] = axes['axis2']
                        row[f"{map_name}_axis3"] = axes['axis3']
                    
                    for i, byte_val in enumerate(decoded['raw_bytes'][:16]):
                        row[f"byte_{i}"] = byte_val
                    
                    csv_writer.writerow(row)
                    csv_file.flush()  # Ensure data is written immediately
                    
            time.sleep(0.01)  # Small delay to avoid CPU spinning
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("Stopping...")
        print(f"✅ Saved {sample_count} samples to {LOG_FILE}")
        print("=" * 60)
    finally:
        if csv_file:
            csv_file.close()
        joystick.close()
        print("Joystick closed.")


if __name__ == "__main__":
    main()

