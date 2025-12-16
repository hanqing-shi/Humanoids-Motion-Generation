import torch
import models
import os
import time
import numpy as np
import argparse
import pandas as pd
from rerun_visualize import RerunJoint
import rerun as rr
from joystick import JoystickController

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='TrajCVAE', help="Model class name in models.py")
    #parser.add_argument("--label-csv", type=str, required=True, help="Path to the label CSV file")
    #parser.add_argument("--traj-csv", type=str, required=True, help="Path to the trajectory CSV file")
    parser.add_argument("--past-lenth", type=int, default=10)
    parser.add_argument('--teacher-forcing-ratio', type=float, default=0)
    parser.add_argument("--seq-len", type=int, default=30)
    return parser.parse_args()

def load_csv_to_tensor(file_path, device, nrows=None):
    if file_path is None:
        return None
    
    df = pd.read_csv(file_path, nrows=nrows) 
    data_np = df.values
    
    data_tensor = torch.from_numpy(data_np).float().to(device)
    
    if data_tensor.ndim == 2:
        data_tensor = data_tensor.unsqueeze(0) 
        
    return data_tensor

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    seq_len = args.seq_len
    ModelClass = getattr(models, args.model)
    
    if args.model == 'TrajCVAE':
        model = ModelClass(traj_dim=36, cond_dim=3, teacher_forcing=args.teacher_forcing_ratio, past_lenth=args.past_lenth).to(device)
        
    print(f"Loaded model: {args.model}")
    
    ckpt_path = f"checkpoints/{args.model}_best_run.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print("✅ Model loaded and ready for inference")
    else:
        print(f"⚠️ Warning: Checkpoint not found at {ckpt_path}")

    model.eval()
    
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    
    #cond_future = load_csv_to_tensor(args.label_csv, device)
    
    #x_init = load_csv_to_tensor(args.traj_csv, device, nrows=args.past_lenth)
    x_init = np.array([
    0.0, 0.0, 0.76,
    0.0, 0.0, 0.0, 1.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0])
    x_init = torch.from_numpy(x_init).float().to(device).unsqueeze(0).unsqueeze(0)  # (1, 1, D_x)

    
    
    #cond_past = torch.zeros((batch_size, args.past_lenth, cond_dim_size)).to(device)
    
    
    controller = JoystickController(motion="walk", deadzone=0.1)
    cond = []
    for i in range(10): 
        cmds = controller.get_cond_commands()
        cond.append(cmds)
    controller.close()
    cond = np.array(cond).reshape(1,-1,3)
    cond_future = torch.from_numpy(cond).float().to(device)
    future_len = cond_future.shape[1]
    cond_dim_size = cond_future.shape[2]
    with torch.no_grad():
        if args.model == 'TrajCVAE':
            # initial context
            cond_past = torch.zeros((1, args.past_lenth, cond_dim_size)).to(device)
            x_past = x_init.repeat(1, args.past_lenth, 1)  # (1, past_lenth, D_x)
            x_start = x_init[:, -1:, :]
            traj_array = []

            for start in range(0, future_len, seq_len):
                end = start + seq_len
                cond = cond_future[:,start:end, :] # (1, T, D_c)
                #cond = torch.zeros_like(cond) # zero condition
                #cond = controller.get_command()
                #
                samples = model.sample(cond_past, cond, x_past, x_start)

                cond_past = cond[:, -args.past_lenth:, :]
                x_past = samples[:, -args.past_lenth:, :]
                x_start = samples[:, -1:, :]

                traj_array.append(samples)  # (1, total_T, D_x)

            traj = torch.cat(traj_array, dim=1).squeeze(0)
            traj = traj.cpu().numpy()
            
            save_path = os.path.join(save_dir, f"{args.model}_1213.csv")
            np.savetxt(save_path, traj, delimiter=",")
            print(f"✅ Sampling complete. Saved to {save_path}")

    rr.init(
        'Reviz', 
        spawn=True
    )
    rr.log('', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    csv_files = f"./results/{args.model}_1213.csv"
    #csv_files = f"{args.file_name}"
    data = np.genfromtxt(csv_files, delimiter=',')
    
    rerun_urdf = RerunJoint('g1')
    print('model type:', 'joint' if data.shape[1] == 36 else 'body')
    for frame_nr in range(data.shape[0]):
        rr.set_time_sequence('frame_nr', frame_nr)
        configuration = data[frame_nr, :]
        rerun_urdf.update(configuration)
        time.sleep(0.03)


if __name__ == "__main__":
    args = parse_cli()
    main(args)