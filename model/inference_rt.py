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
    parser.add_argument("--past-lenth", type=int, default=10)
    parser.add_argument("--inference-step", type=int, default=300)
    return parser.parse_args()

def main(args):
    # model init
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    ModelClass = getattr(models, args.model)
    
    if args.model == 'TrajCVAE':
        model = ModelClass(traj_dim=36, cond_dim=3, teacher_forcing=0, past_lenth=args.past_lenth).to(device)
        
    print(f"Loaded model: {args.model}")
    
    ckpt_path = f"checkpoints/{args.model}_best_walk.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print("✅ Model loaded and ready for inference")
    else:
        print(f"⚠️ Warning: Checkpoint not found at {ckpt_path}")

    model.eval()
    
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    
    # initial configuration
    x_init = np.array([
    0.0, 0.0, 0.76,
    0.0, 0.0, 0.0, 1.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0])
    x_init = torch.from_numpy(x_init).float().to(device).unsqueeze(0).unsqueeze(0)  # (1, 1, D_x)

    # joystick controller init
    controller = JoystickController(motion="walk", deadzone=0.1)
    assert controller.joystick is not None, "Joystick not connected. Please connect a joystick and try again."

    # visualizer init
    rr.init(
        'Reviz', 
        spawn=True
    )
    rr.log('', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rerun_urdf = RerunJoint('g1')


    with torch.no_grad():
        if args.model == 'TrajCVAE':
            # initial context
            cond_past = torch.zeros((1, args.past_lenth, 3)).to(device)
            x_past = x_init.repeat(1, args.past_lenth, 1)  # (1, past_lenth, D_x)
            x_start = x_init[:, -1:, :]
            traj_array = []

            for frame in range(args.inference_step):
                
                # read command
                cmd = controller.get_cond_commands(steps=20, freq=30.0).reshape(1,-1,3)
                cond = torch.from_numpy(cmd).float().to(device)

                # inference
                samples = model.sample(cond_past, cond, x_past, x_start)
                # visualization
                for step in range(samples.shape[1]):
                    rr.set_time_sequence('frame_nr', frame*20 + step)
                    configuration = samples[-1, step, :].detach().cpu().numpy()
                    rerun_urdf.update(configuration) 
                    time.sleep(0.03)  # 30Hz

                # update past information
                cond_past = torch.cat([cond_past, cond], dim=1)
                cond_past = cond_past[:, -args.past_lenth:, :]

                new_state = samples[:, -1:, :] # (1, 1, D_x)
                x_past = torch.cat([x_past, new_state], dim=1)
                x_past = x_past[:, -args.past_lenth:, :]
                x_start = new_state

                traj_array.append(samples)  # (1, total_T, D_x)
                
            traj = torch.cat(traj_array, dim=1).squeeze(0)
            traj = traj.cpu().numpy()
            controller.close()

            # save results
            save_path = os.path.join(save_dir, f"{args.model}_1213.csv")
            np.savetxt(save_path, traj, delimiter=",")
            print(f"✅ Sampling complete. Saved to {save_path}")

if __name__ == "__main__":
    args = parse_cli()
    main(args)