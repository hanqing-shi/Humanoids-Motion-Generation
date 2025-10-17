import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import MotionDataset
import models


def get_dataloader(batch_size, num_workers=4, shuffle=True):
    dataset = MotionDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)


def train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Initialize dataloader
    loader = get_dataloader(args.batch_size, args.num_workers)
    
    # Load model 
    if not hasattr(models, args.model):
        raise ValueError(f"models.py does not define a class named '{args.model}'")
    ModelClass = getattr(models, args.model)
    model = ModelClass().to(device)
    print(f"Loaded model: {args.model}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Checkpoint setup
    ckpt_dir = Path(args.save_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = ckpt_dir / "last.pt"
    start_epoch = 0

    # Resume if needed
    if args.resume and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optim"])
        start_epoch = ck["epoch"] + 1
        print(f"🔁 Resumed from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        log_interval = 20
        save_freq = 5
        for i, batch in enumerate(loader):
            state, label = batch
            x, y = state.to(device), label.to(device)
            optimizer.zero_grad()

            # Forward pass
            pred, loss = model(x, y)
            # Backpropagation
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % log_interval == 0:
                print(f"Epoch {epoch} [{i}/{len(loader)}] Loss: {loss.item():.6f}")

        avg_loss = total_loss / len(loader)
        print(f"✅ Epoch {epoch} finished in {time.time()-t0:.1f}s | Avg Loss: {avg_loss:.6f}")

        # Save checkpoints
        torch.save(
            {"model": model.state_dict(), "optim": optimizer.state_dict(), "epoch": epoch},
            last_ckpt,
        )
        if (epoch + 1) % save_freq == 0:
            torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch+1}.pt")

    print(f"🎉 Training complete! Checkpoints saved to: {ckpt_dir}")


def parse_cli():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="General PyTorch Training Script")
    parser.add_argument("--model", required=True, help="Model class name in models.py (e.g. TrajCVAE)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--save-dir", default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training from last checkpoint")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli()
    train(args)
