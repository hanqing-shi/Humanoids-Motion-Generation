import argparse
import time
import yaml
import os
import torch
import numpy as np
import threading
import collections
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
import rerun as rr
import rerun.blueprint as rrb
import models
from joystick import JoystickController
from dataset import JointDataset
from rerun_visualize import RerunJoint

# --- Shared Globals ---
# Thread-safe buffer for data exchange between Joystick (Main) and Inference (Worker)
command_buffer = collections.deque(maxlen=20)
buffer_lock = threading.Lock()
running = True 

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    return parser.parse_args()

def inference_worker(conf, model, device, data_stats, norm_mask, x_init, x_init_phys, rerun_urdf):
    """
    Worker thread handles heavy model inference and Rerun visualization.
    """
    global running, command_buffer
    
    data_min, data_range = data_stats
    past_len = conf['model']['past_lenth']
    inference_steps = conf['inference']['step']
    vis_dt = 1.0 / conf['inference']['vis_freq']
    quat_smoothing_alpha = conf['inference'].get('quat_smoothing_alpha', 0.5)
    
    # Initial Context
    cond_past = torch.zeros((1, past_len, 3)).to(device)
    x_past = x_init.repeat(1, past_len, 1)
    x_start = x_init[:, -1:, :]
    
    global_root_pos = np.array([0.0, 0.0])
    prev_quat = None

    # Repeat the first frame 10 times.
    # 1. Prepare a single-frame sample (shape: 1 x D).
    init_traj_frame = x_init_phys.reshape(1, -1) 
    init_cmd_frame = np.zeros((1, 3), dtype=np.float32)

    # 2. Store 10 repeated copies so concatenation starts with 10 initial frames.
    traj_history = [init_traj_frame for _ in range(10)]
    cmd_history = [init_cmd_frame for _ in range(10)]
    
    print("🚀 Inference Thread Started...")

    with torch.no_grad():
        for frame in range(inference_steps):
            if not running: break

            # Thread-safe command fetch
            with buffer_lock:
                # Pad buffer with zeros if empty or incomplete
                while len(command_buffer) < 20:
                    command_buffer.append(np.zeros(3, dtype=np.float32))
                
                cmd_seq = np.array(command_buffer, dtype=np.float32) 
            
            # Record the current command sequence (20, 3).
            cmd_history.append(cmd_seq)

            # Run Inference
            cmd_tensor = torch.from_numpy(cmd_seq).float().to(device).unsqueeze(0)
            samples_norm = model.sample(cond_past, cmd_tensor, x_past, x_start)

            # Denormalize output
            samples_phys = samples_norm.clone()
            samples_phys[..., norm_mask] = (samples_norm[..., norm_mask] + 1) / 2 * data_range[norm_mask] + data_min[norm_mask]
            samples_phys[..., 3:7] = F.normalize(samples_phys[..., 3:7], dim=-1)

            trajectory = samples_phys[0].cpu().numpy()

            # Visualization Loop
            for t in range(trajectory.shape[0]):
                if not running: break

                local_vel = trajectory[t, :2]
                quat_xyzw = trajectory[t, 3:7]

                if prev_quat is not None:
                    if np.dot(prev_quat, quat_xyzw) < 0:
                        quat_xyzw = -quat_xyzw
                    quat_xyzw = (1.0 - quat_smoothing_alpha) * prev_quat + quat_smoothing_alpha * quat_xyzw
                    quat_xyzw /= np.linalg.norm(quat_xyzw)

                trajectory[t, 3:7] = quat_xyzw
                prev_quat = quat_xyzw.copy()
                
                rot = R.from_quat(quat_xyzw)
                global_vel_3d = rot.apply(np.array([local_vel[0], local_vel[1], 0.0]))
                global_root_pos += global_vel_3d[:2] * vis_dt

                trajectory[t, 0] = global_root_pos[0]
                trajectory[t, 1] = global_root_pos[1]
                
                rr.set_time(timeline='frame_nr', sequence=frame * 20 + t)
                rerun_urdf.update(trajectory[t, :])
                time.sleep(vis_dt) 

            # Update Autoregressive State
            cond_past = torch.cat([cond_past, cmd_tensor], dim=1)[:, -past_len:, :]
            x_past = torch.cat([x_past, samples_norm], dim=1)[:, -past_len:, :]
            x_start = samples_norm[:, -1:, :]
            
            # Record the current trajectory (20, 36).
            traj_history.append(trajectory)

    # Save Results
    if traj_history:
        # 1. Save trajectory.
        full_traj = np.concatenate(traj_history, axis=0)
        traj_save_path = os.path.join(conf['save_dir'], f"{conf['model']['name']}_rt_result.csv")
        np.savetxt(traj_save_path, full_traj, delimiter=",")
        
        # 2. Save commands.
        full_cmd = np.concatenate(cmd_history, axis=0)
        cmd_save_path = os.path.join(conf['save_dir'], f"{conf['model']['name']}_rt_command.csv")
        np.savetxt(cmd_save_path, full_cmd, delimiter=",")
        
        print(f"✅ Saved Trajectory to {traj_save_path} (Shape: {full_traj.shape})")
        print(f"✅ Saved Commands to {cmd_save_path} (Shape: {full_cmd.shape})")
    
    print("👋 Inference Thread Finished.")
    running = False 


def main(args):
    global running
    
    # Load Config
    with open(args.config, 'r') as f:
        conf = yaml.safe_load(f)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load Scaler
    scaler_path = os.path.join(conf['save_dir'], "scaler.pt")
    scaler = torch.load(scaler_path, map_location=device)
    data_min = scaler['min']
    data_range = scaler['range']
    traj_dim = data_min.shape[0]
    cond_dim = conf['model'].get('cond_dim', 3)

    col_names = JointDataset.DEFAULT_ROOT_STATE + JointDataset.DEFAULT_JOINT_NAMES
    norm_mask = torch.ones(traj_dim, dtype=torch.bool, device=device)
    for i, name in enumerate(col_names):
        if 'root_q' in name: norm_mask[i] = False

    # Initialize Model
    model_name = conf['model']['name']
    ModelClass = getattr(models, model_name)
    model = ModelClass(traj_dim=traj_dim, cond_dim=cond_dim, teacher_forcing=0, past_lenth=conf['model']['past_lenth']).to(device)
    
    ckpt_path = os.path.join(conf['save_dir'], f"{model_name}_best.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    model.eval()

    # Init Initial State (Physical)
    x_init_phys = np.array([
        0.0, 0.0, 0.76, 0.0, 0.0, 0.0, 1.0, 
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0, -0.312, 0.0, 0.0, 0.669, -0.363, 0.0, 0.0, 0.0, 0.0, 
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0, 0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0
    ], dtype=np.float32)
    x_init_tensor = torch.from_numpy(x_init_phys).to(device)
    x_init_norm = x_init_tensor.clone()
    x_init_norm[norm_mask] = 2 * (x_init_tensor[norm_mask] - data_min[norm_mask]) / data_range[norm_mask] - 1
    x_init = x_init_norm.unsqueeze(0).unsqueeze(0)

    # Init Rerun
    rr.init('Rviz', spawn=True)
    rr.send_blueprint(
    rrb.Blueprint(
        rrb.Spatial3DView(origin="/", name="3D View"),
        collapse_panels=True,
    )
)

    rr.log('', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rerun_urdf = RerunJoint('g1')

    # Initialize Joystick
    try:
        motion_name = conf['inference']['motions']
        controller = JoystickController(motion=motion_name, deadzone=conf['inference']['joystick_deadzone'])
        
        # Pre-fill buffer
        for _ in range(20):
            command_buffer.append(np.zeros(3, dtype=np.float32))
            
    except Exception as e:
        print(f"Joystick Error: {e}")
        return

    # Start Inference Worker Thread
    t = threading.Thread(
        target=inference_worker, 
        args=(conf, model, device, (data_min, data_range), norm_mask, x_init, x_init_phys, rerun_urdf),
        daemon=True
    )
    t.start()

    # Main Loop: Polling Joystick
    print("🎮 Main Thread: Polling Joystick...")
    try:
        dt = 1.0 / 30.0
        while running and t.is_alive():
            start_t = time.perf_counter()
            
            # Poll hardware input
            cmd = controller.get_command()
            
            # Push to shared buffer
            with buffer_lock:
                command_buffer.append(cmd)
            
            # Maintain sampling rate
            elapsed = time.perf_counter() - start_t
            time.sleep(max(0, dt - elapsed))
            
    except KeyboardInterrupt:
        print("\nStopping...")
        running = False
    finally:
        controller.close()
        t.join()

if __name__ == "__main__":
    args = parse_cli()
    main(args)
