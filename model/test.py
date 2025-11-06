import torch
import models
from dataset import MotionDataset, JointDataset
from torch.utils.data import DataLoader
import os
import numpy as np
import argparse

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='MlpCVAE', help="Model class name in models.py")
    return parser.parse_args()

def main(args):
    if args.model == 'TrajCVAE':
        val_dataset = MotionDataset(
            data_dir = './dataset/data_test_feature', 
            label_dir = './dataset/data_test_label', 
            seq_len=30, 
            motions = ['walk'],  # list of motion subfolders
            columns=("pos", "ori"),
            stride=30,
            transform=None
        )
    elif args.model == 'MlpCVAE':
        val_dataset = JointDataset(
            data_dir = './dataset/data_test_joint', 
            label_dir = './dataset/data_test_label', 
            seq_len=30, 
            motions = ['walk'],  # list of motion subfolders
            stride=30,
            transform=None
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    frame_size, traj_dim,  = val_loader.dataset[0][0].shape
    cond_dim = val_loader.dataset[0][1].shape[-1]
    ModelClass = getattr(models, args.model)
    if args.model == 'TrajCVAE':
        model = ModelClass(traj_dim, cond_dim).to(device)
    elif args.model == 'MlpCVAE':
        model = ModelClass(
            frame_size,
            latent_size = 64,
            condition_size = frame_size * cond_dim + traj_dim,
            feature_size = traj_dim,
            hidden_size=128,
        ).to(device)
    
    print(f"Loaded model: {args.model}")

    # best or last
    ckpt = torch.load(f"checkpoints/{args.model}_best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("✅ Model loaded and ready for inference")
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)

    n_samples = 1  # number of random samples per input
    max_batches = 3  # limit how many batches to visualize/save

    criterion = torch.nn.MSELoss(reduction='mean')
    
    with torch.no_grad():
        for batch_idx, (state, label) in enumerate(val_loader):
            # Each batch contains:
            #   state: (B, T, D_x) - target future sequence
            #   label: (B, T, D_c) - condition sequence (e.g., velocities)
            if args.model == 'TrajCVAE':
                cond_seq = label.to(device)  # condition input
                # N = 30
                # cond_seq = torch.zeros(32 ,N, 3).to(device)  # zero condition for testing
                #cond_seq[:,:,2] *= -1
                x0 = state[:, 0, :].to(device)  # initial state (first frame)
                # Generate trajectories from the prior
                samples = model.sample(cond_seq, x0, n_samples=n_samples)  # (B, n_samples, T, D_x)
            elif args.model == 'MlpCVAE':
                z = torch.randn(32, 64, device=device)
                c = torch.cat([state[:,0,:].flatten(1), label.flatten(1)], dim=-1).to(device)  # concatenate current pose and future velocity label
                c = torch.zeros_like(c)
                # Generate trajectories from the prior
                samples = model.sample(z, c)  # (B, T, D_x)
                B, T, D = samples.shape
            
            # B, N, T, D = samples.shape
            
            # Predict trajectory (use reparameterized z)
            # y_hat, loss, metrics = model(state, cond_seq)  # Forward pass in eval mode
            # mse_loss = criterion(y_hat, samples.squeeze(1)).item()
            # print(f"Batch {batch_idx}: MSE = {mse_loss:.6f}, Recon = {metrics['recon']:.6f}, KL = {metrics['kl']:.6f}")
            # Save results to disk
            traj_list = []
            for i in range(B): # B seconds if seq_len=30
                traj = samples[i].cpu().numpy()
                traj_list.append(traj)
                # save_path = os.path.join(save_dir, f"sample_{batch_idx:03d}_b{i}_n{n}.csv")
                # np.savetxt(save_path, traj, delimiter=",")
                # print(f"💾 Saved {save_path}")
                traj_array = np.stack(traj_list, axis=0)  # (B, T, D_x)
            traj_array = traj_array.reshape(-1, D)
            save_path = os.path.join(save_dir, f"joint_merged_sample_{batch_idx:03d}_b{i}.csv")
            np.savetxt(save_path, traj_array,delimiter=",")
            
            if batch_idx >= max_batches - 1:
                break

    print("✅ Sampling complete.")

if __name__ == "__main__":
    args = parse_cli()
    main(args)