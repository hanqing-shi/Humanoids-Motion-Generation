import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import MotionDataset, JointDataset
import models

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='TrajCVAE', help="Model class name in models.py")
    parser.add_argument("--dataset", default='feature', help="joint or feature")
    parser.add_argument("--teacher-forcing-ratio", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--resume", action="store_true", default=False)
    return parser.parse_args()

def kl_divergence(mu, logvar):
    """KL( q(z|.) || N(0,I) ) per-sample"""
    return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=-1)

def get_dataloader(batch_size, dataset, num_workers=4, shuffle=True):
    if dataset == 'feature':
        dataset = MotionDataset( 
                    data_dir = './dataset/data_feature', 
                    label_dir = './dataset/data_label', 
                    seq_len=30, # choose length of each sequence
                    motions = ['walk'],
                    columns=("pos", "ori"),
                    stride=10,
                    transform=None)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    if dataset == 'joint':
        dataset = JointDataset( 
                    data_dir = './dataset/data_joint', 
                    label_dir = './dataset/data_label', 
                    seq_len=30, 
                    motions = ['walk'],
                    stride=10,
                    transform=None)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

def train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Initialize dataloader
    loader = get_dataloader(args.batch_size, args.dataset, num_workers=4)
    
    # Load model 
    if not hasattr(models, args.model):
        raise ValueError(f"models.py does not define a class named '{args.model}'")
    
    frame_size, traj_dim,  = loader.dataset[0][0].shape
    cond_dim = loader.dataset[0][1].shape[-1]
    ModelClass = getattr(models, args.model)
    if args.model == 'TrajCVAE':
        model = ModelClass(
            traj_dim, 
            cond_dim,
            teacher_forcing=args.teacher_forcing_ratio,
            past_lenth=10
        ).to(device)
    elif args.model == 'MlpCVAE':
        model = ModelClass(
            frame_size,
            latent_size = 64,
            condition_size = frame_size * cond_dim + traj_dim,
            feature_size = traj_dim,
            hidden_size=128,
        ).to(device)
    print(f"Loaded model: {args.model}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Checkpoint setup
    ckpt_dir = Path(args.save_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = ckpt_dir / f"{args.model}_last.pt"
    start_epoch = 0

    # Resume if needed
    if args.resume and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optim"])
        start_epoch = ck["epoch"] + 1
        print(f"🔁 Resumed from epoch {start_epoch}")

    # Training loop
    best_loss = float("inf")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        log_interval = 20
        save_freq = 5
        for i, batch in enumerate(loader):
            state, label = batch
            x, y = state.to(device), label.to(device)
            if args.model == 'TrajCVAE':
                c = y
            elif args.model == 'MlpCVAE':
                c = torch.cat([x[:,0,:].flatten(1), y.flatten(1)], dim=-1) # concatenate current pose and future velocity label

            optimizer.zero_grad()

            # Forward pass
            pred, mu, logvar = model(x, c)

            kl_loss = kl_divergence(mu, logvar).mean()
            recon_loss = F.mse_loss(state, pred, reduction="none").sum(dim=(1, 2)).mean()
            beta = 1.0
            loss = recon_loss + beta * kl_loss
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % log_interval == 0:
                print(f"Epoch {epoch} [{i}/{len(loader)}] Loss: {loss.item():.6f} | Recon: {recon_loss.item():.6f} | KL: {kl_loss.item():.6f}")
                
        print(len(loader))
        avg_loss = total_loss / len(loader)
        print(f"✅ Epoch {epoch} finished in {time.time()-t0:.1f}s | Avg Loss: {avg_loss:.6f}")

        # Save checkpoints
        torch.save(
            {"model": model.state_dict(), "optim": optimizer.state_dict(), "epoch": epoch},
            last_ckpt,
        )

        # 💾 Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_ckpt = ckpt_dir / f"{args.model}_best.pt"
            torch.save(
                {"model": model.state_dict(), "optim": optimizer.state_dict(), "epoch": epoch},
                best_ckpt,
            )
            print(f"🌟 New best model saved at epoch {epoch} with loss {best_loss:.6f}")

        if (epoch + 1) % save_freq == 0:
            torch.save({"model": model.state_dict(), "optim": optimizer.state_dict(), "epoch": epoch}, ckpt_dir / f"{args.model}_epoch_{epoch+1}.pt")

    print(f"🎉 Training complete! Checkpoints saved to: {ckpt_dir}")

if __name__ == "__main__":
    args = parse_cli()
    train(args)
