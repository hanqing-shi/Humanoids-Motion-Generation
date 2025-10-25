import torch
import models
from dataset import MotionDataset
from torch.utils.data import DataLoader
import os
import numpy as np

def main():
    # ============================================
    # Load Validation Dataset
    # ============================================
    val_dataset = MotionDataset(
        data_dir = './data_feature', 
        label_dir = './data_label', 
        seq_len=30, 
        motions = ['walk'],  # list of motion subfolders
        columns=("pos", "ori", "vel", "angvel"),
        stride=10,
        transform=None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    traj_dim, cond_dim = val_loader.dataset[0][0].shape[-1], val_loader.dataset[0][1].shape[-1]
    # ============================================
    # 2️⃣ Load Model and Checkpoint
    # ============================================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    # Define model dimensions based on dataset column structure
    model = models.TrajCVAE(
        traj_dim=traj_dim,
        cond_dim=cond_dim # condition dimension (velocity + ang. velocity)
    ).to(device)

    # Load trained checkpoint
    ckpt = torch.load("checkpoints/best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("✅ Model loaded and ready for inference")

    # ============================================
    # 3️⃣ Sampling (Inference)
    # ============================================
    save_dir = "results_generated"
    os.makedirs(save_dir, exist_ok=True)

    n_samples = 1  # number of random samples per input
    max_batches = 3  # limit how many batches to visualize/save

    criterion = torch.nn.MSELoss(reduction='mean')

    with torch.no_grad():
        for batch_idx, (state, label) in enumerate(val_loader):
            # Each batch contains:
            #   state: (B, T, D_x) - target future sequence
            #   label: (B, T, D_c) - condition sequence (e.g., velocities)

            cond_seq = label.to(device)  # condition input
            # N = 300
            # cond_seq = torch.zeros(32 ,N, 3).to(device)  # zero condition for testing
            # cond_seq[:,1] = 0.5
            x0 = state[:, 0, :].to(device)  # initial state (first frame)

            # Generate trajectories from the prior
            samples = model.sample(cond_seq, x0, n_samples=n_samples)  # (B, n_samples, T, D_x)
            B, N, T, D = samples.shape
            
            # Predict trajectory (use reparameterized z)
            # y_hat, loss, metrics = model(state, cond_seq)  # Forward pass in eval mode
            # mse_loss = criterion(y_hat, state).item()
            # print(f"Batch {batch_idx}: MSE = {mse_loss:.6f}, Recon = {metrics['recon']:.6f}, KL = {metrics['kl']:.6f}")
            # Save results to disk
            for i in range(min(3, B)):  # only save first few examples per batch
                for n in range(N):
                    traj = samples[i, n].cpu().numpy()
                    save_path = os.path.join(save_dir, f"sample_{batch_idx:03d}_b{i}_n{n}.csv")
                    np.savetxt(save_path, traj, delimiter=",")
                    print(f"💾 Saved {save_path}")

            if batch_idx >= max_batches - 1:
                break

    print("✅ Sampling complete.")

if __name__ == "__main__":
    main()