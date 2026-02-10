#!/usr/bin/env python3
"""
MuJoCo CSV playback script for Unitree G1.
Reads segments from segments_sangwoo_251012.csv and plays back motion data in MuJoCo.
"""

import os
import csv
import time
import argparse
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

# ============================================================
# CONFIG
# ============================================================

FPS = 60.0
DT = 1.0 / FPS

# Path to MuJoCo model
MUJOCO_MENAGERIE_DIR = Path(__file__).resolve().parents[2] / "mujoco_menagerie"
G1_XML_PATH = MUJOCO_MENAGERIE_DIR / "unitree_g1" / "g1.xml"
G1_SCENE_XML_PATH = MUJOCO_MENAGERIE_DIR / "unitree_g1" / "scene.xml"

# Default CSV directory (adjust as needed)
DEFAULT_CSV_DIR = Path(__file__).parent


# ============================================================
# CSV LOADING
# ============================================================

def load_motion_csv(csv_path, start_row=0, end_row=None):
    """
    Load motion data from CSV file.
    Expected format: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z, joint_1, ..., joint_N]
    
    Args:
        csv_path: Path to CSV file
        start_row: Starting row index (0-indexed)
        end_row: Ending row index (exclusive, None = end of file)
    
    Returns:
        numpy array of shape (T, D) where T is number of frames, D is dimension
    """
    data = np.loadtxt(csv_path, delimiter=',')
    
    if data.ndim == 1:
        data = data[None, :]
    
    if end_row is None:
        end_row = len(data)
    
    # Extract segment
    segment = data[start_row:end_row]
    
    print(f"Loaded {len(segment)} frames from {csv_path} (rows {start_row} to {end_row-1})")
    print(f"Data shape: {segment.shape}")
    
    return segment


def load_segments_csv(segments_csv_path):
    """
    Load segments specification from CSV.
    Expected format: file,start,end,out
    
    Returns:
        List of dicts with keys: 'file', 'start', 'end', 'out'
    """
    segments = []
    
    with open(segments_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get('file') or not row.get('start') or not row.get('end'):
                continue
            
            segments.append({
                'file': row['file'].strip(),
                'start': int(row['start']),
                'end': int(row['end']),
                'out': row.get('out', '').strip() if row.get('out') else None
            })
    
    return segments


# ============================================================
# MUJOCO PLAYBACK
# ============================================================

def setup_robot_initial_pose(model, data, free_joint_id, motion_data):
    """
    Set initial robot pose from first frame of motion data.
    motion_data[0] should be [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z, ...joints...]
    """
    if free_joint_id is None:
        print("⚠️  Warning: No free joint found. Cannot set base pose.")
        return
    
    jnt_qposadr = model.jnt_qposadr[free_joint_id]
    
    # Set base position and orientation
    if motion_data.shape[1] >= 7:
        # Position (x, y, z)
        data.qpos[jnt_qposadr:jnt_qposadr + 3] = motion_data[0, :3]
        
        # Quaternion (w, x, y, z) - MuJoCo uses (w, x, y, z) format
        quat = motion_data[0, 3:7]
        quat_norm = np.linalg.norm(quat)
        if quat_norm > 1e-6:
            quat = quat / quat_norm
        data.qpos[jnt_qposadr + 3:jnt_qposadr + 7] = quat
    
    # Set joint positions (skip first 7 elements which are base pose)
    if motion_data.shape[1] > 7:
        joint_data = motion_data[0, 7:]
        n_joints = min(len(joint_data), model.nq - 7)
        
        # Find joint indices (skip free joint)
        joint_idx = 0
        for i in range(model.njnt):
            if i == free_joint_id:
                continue
            if joint_idx >= n_joints:
                break
            
            jnt_qposadr_joint = model.jnt_qposadr[i]
            jnt_type = model.jnt_type[i]
            
            if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                continue
            
            # Set joint position
            if jnt_type == mujoco.mjtJoint.mjJNT_HINGE or jnt_type == mujoco.mjtJoint.mjJNT_SLIDE:
                data.qpos[jnt_qposadr_joint] = joint_data[joint_idx]
                joint_idx += 1
            elif jnt_type == mujoco.mjtJoint.mjJNT_BALL:
                # Ball joint uses quaternion (4 values)
                if joint_idx + 3 < n_joints:
                    # Assume joint_data has euler angles, convert to quaternion
                    # For now, just set to identity
                    data.qpos[jnt_qposadr_joint:jnt_qposadr_joint + 4] = [1, 0, 0, 0]
                    joint_idx += 3
    
    # Forward kinematics
    mujoco.mj_forward(model, data)
    
    # Adjust base height so feet are on ground
    if free_joint_id is not None:
        jnt_qposadr = model.jnt_qposadr[free_joint_id]
        
        # Find foot sites
        left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
        
        if left_foot_id >= 0 and right_foot_id >= 0:
            left_foot_pos = data.site_xpos[left_foot_id]
            right_foot_pos = data.site_xpos[right_foot_id]
            
            min_foot_z = min(left_foot_pos[2], right_foot_pos[2]) - 0.03
            current_pelvis_z = data.qpos[jnt_qposadr + 2]
            adjustment = 0.0 - min_foot_z
            new_pelvis_z = current_pelvis_z + adjustment
            
            data.qpos[jnt_qposadr + 2] = new_pelvis_z
            print(f"Adjusted pelvis z from {current_pelvis_z:.3f} to {new_pelvis_z:.3f} m")
        else:
            # Fallback: use approximate height
            data.qpos[jnt_qposadr + 2] = 0.165
            print("⚠️  Could not find foot sites, using approximate height")
        
        mujoco.mj_forward(model, data)


def apply_motion_frame(model, data, free_joint_id, frame_data):
    """
    Apply a single frame of motion data to the robot.
    frame_data: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z, ...joints...]
    """
    if free_joint_id is None:
        return
    
    jnt_qposadr = model.jnt_qposadr[free_joint_id]
    
    # Update base position and orientation
    if len(frame_data) >= 7:
        # Position
        data.qpos[jnt_qposadr:jnt_qposadr + 3] = frame_data[:3]
        
        # Quaternion
        quat = frame_data[3:7]
        quat_norm = np.linalg.norm(quat)
        if quat_norm > 1e-6:
            quat = quat / quat_norm
        data.qpos[jnt_qposadr + 3:jnt_qposadr + 7] = quat
    
    # Update joint positions
    if len(frame_data) > 7:
        joint_data = frame_data[7:]
        n_joints = min(len(joint_data), model.nq - 7)
        
        joint_idx = 0
        for i in range(model.njnt):
            if i == free_joint_id:
                continue
            if joint_idx >= n_joints:
                break
            
            jnt_qposadr_joint = model.jnt_qposadr[i]
            jnt_type = model.jnt_type[i]
            
            if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                continue
            
            if jnt_type == mujoco.mjtJoint.mjJNT_HINGE or jnt_type == mujoco.mjtJoint.mjJNT_SLIDE:
                data.qpos[jnt_qposadr_joint] = joint_data[joint_idx]
                joint_idx += 1
            elif jnt_type == mujoco.mjtJoint.mjJNT_BALL:
                # Ball joint - skip for now or handle separately
                if joint_idx + 3 < n_joints:
                    joint_idx += 3
    
    # Forward kinematics
    mujoco.mj_forward(model, data)


def playback_segment(model, data, free_joint_id, motion_data, segment_name="", playback_speed=1.0):
    """
    Play back a motion segment in MuJoCo viewer.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        free_joint_id: Free joint ID for base
        motion_data: Motion data array (T, D)
        segment_name: Name of segment for display
        playback_speed: Speed multiplier (1.0 = normal, 2.0 = 2x speed, etc.)
    """
    T = len(motion_data)
    frame_dt = DT / playback_speed
    
    print(f"\nPlaying segment: {segment_name}")
    print(f"  Frames: {T}")
    print(f"  Duration: {T * frame_dt:.2f} seconds")
    print(f"  Playback speed: {playback_speed}x")
    print("  Press Ctrl+C to stop or skip to next segment\n")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set initial camera
        viewer.cam.lookat[:] = [0, 0, 0.8]
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -20
        
        frame_idx = 0
        start_time = time.time()
        
        try:
            while viewer.is_running() and frame_idx < T:
                # Apply current frame
                apply_motion_frame(model, data, free_joint_id, motion_data[frame_idx])
                
                # Update camera to follow robot
                if free_joint_id is not None:
                    jnt_qposadr = model.jnt_qposadr[free_joint_id]
                    base_pos = data.qpos[jnt_qposadr:jnt_qposadr + 3]
                    viewer.cam.lookat[:] = base_pos.copy()
                    viewer.cam.lookat[2] += 0.8
                
                # Sync viewer
                viewer.sync()
                
                # Print progress every second
                if frame_idx % int(FPS) == 0:
                    elapsed = time.time() - start_time
                    progress = (frame_idx + 1) / T * 100
                    print(f"  Frame {frame_idx+1}/{T} ({progress:.1f}%) - {elapsed:.1f}s", end='\r')
                
                frame_idx += 1
                time.sleep(frame_dt)
            
            print(f"\n✅ Segment completed: {segment_name}")
            
        except KeyboardInterrupt:
            print(f"\n⏸️  Segment interrupted: {segment_name}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Play back CSV motion data in MuJoCo")
    parser.add_argument(
        "--segments_csv",
        type=str,
        default="segments_sangwoo_251012.csv",
        help="Path to segments CSV file"
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        default=None,
        help="Directory containing motion CSV files (default: same as segments_csv)"
    )
    parser.add_argument(
        "--segment_idx",
        type=int,
        default=None,
        help="Play specific segment index (0-indexed, default: play all)"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (default: 1.0)"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["scene", "g1"],
        default="scene",
        help="MuJoCo model to use: 'scene' (with grid) or 'g1' (plain)"
    )
    
    args = parser.parse_args()
    
    # Load segments
    segments_csv_path = Path(args.segments_csv)
    if not segments_csv_path.exists():
        segments_csv_path = DEFAULT_CSV_DIR / args.segments_csv
    
    if not segments_csv_path.exists():
        print(f"❌ Error: Segments CSV not found at {segments_csv_path}")
        return
    
    segments = load_segments_csv(segments_csv_path)
    print(f"\n✅ Loaded {len(segments)} segments from {segments_csv_path}")
    
    if len(segments) == 0:
        print("❌ No segments found in CSV file")
        return
    
    # Determine CSV directory - try multiple possible locations
    if args.csv_dir:
        csv_dir = Path(args.csv_dir)
    else:
        csv_dir = segments_csv_path.parent
    
    # Try common subdirectories - prioritize retarget_out_vel in script directory
    script_dir = Path(__file__).parent
    possible_dirs = [
        script_dir / "retarget_out_vel",  # First priority: retarget_out_vel in script directory
        csv_dir / "retarget_out_vel",  # Then try retarget_out_vel in csv_dir
        script_dir / "g1",  # Then g1 in script directory
        csv_dir / "g1",
        csv_dir,
        csv_dir.parent / "retarget_out_vel",
        csv_dir.parent / "g1",
        csv_dir.parent / "g1_retargeted_dataset",
    ]
    
    csv_dir_found = None
    for possible_dir in possible_dirs:
        if possible_dir.exists() and possible_dir.is_dir():
            csv_dir_found = possible_dir
            print(f"📁 Using CSV directory: {csv_dir_found}")
            break
    
    if csv_dir_found is None:
        print(f"⚠️  Warning: CSV directory not found, will try: {csv_dir}")
        csv_dir_found = csv_dir
    
    # Load MuJoCo model
    if args.model == "scene":
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
    
    # Disable gravity for playback
    model.opt.gravity[:] = 0.0
    
    # Determine which segments to play
    if args.segment_idx is not None:
        if args.segment_idx < 0 or args.segment_idx >= len(segments):
            print(f"❌ Error: Segment index {args.segment_idx} out of range (0-{len(segments)-1})")
            return
        segments_to_play = [segments[args.segment_idx]]
    else:
        segments_to_play = segments
    
    # Play each segment
    for seg_idx, segment in enumerate(segments_to_play):
        csv_file = segment['file']
        start_row = segment['start']
        end_row = segment['end']
        segment_name = segment.get('out') or f"{csv_file} [{start_row}:{end_row}]"
        
        # Try multiple possible paths for CSV file
        csv_path = None
        csv_file_base = csv_file.replace('.csv', '')
        
        # First try exact match
        possible_paths = [
            csv_dir_found / csv_file,
        ]
        
        # Then try variations with common suffixes (especially for retarget_out_vel)
        if '_vel' not in csv_file:
            possible_paths.extend([
                csv_dir_found / f"{csv_file_base}_vel.csv",
                csv_dir_found / f"{csv_file_base}_1_vel.csv",
                csv_dir_found / f"{csv_file_base}_mid_1_vel.csv",
                csv_dir_found / f"{csv_file_base}_mid.csv",
            ])
        
        # Also try nested and other locations
        possible_paths.extend([
            csv_dir_found / csv_file_base / csv_file,  # nested
            Path(csv_file),  # absolute path
            csv_dir_found.parent / csv_file,  # parent directory
        ])
        
        for path in possible_paths:
            if path.exists() and path.is_file():
                csv_path = path
                print(f"📄 Found CSV file: {csv_path}")
                break
        
        if csv_path is None:
            print(f"❌ Error: Motion CSV not found. Tried:")
            for path in possible_paths:
                print(f"     - {path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Segment {seg_idx + 1}/{len(segments_to_play)}: {segment_name}")
        print(f"{'='*60}")
        
        # Load motion data
        try:
            motion_data = load_motion_csv(csv_path, start_row, end_row)
        except Exception as e:
            print(f"❌ Error loading motion data: {e}")
            continue
        
        # Set initial pose
        setup_robot_initial_pose(model, data, free_joint_id, motion_data)
        
        # Play back
        try:
            playback_segment(model, data, free_joint_id, motion_data, segment_name, args.speed)
        except KeyboardInterrupt:
            print("\n\n⏹️  Playback stopped by user")
            break
    
    print("\n✅ All segments completed!")


if __name__ == "__main__":
    main()

