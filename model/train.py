import argparse
import time
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import JointDataset
import models

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--resume", action="store_true", help="Override config to resume training")
    return parser.parse_args()

def get_dataloader(conf, split='train'):
    if split == 'train':
        dataset = JointDataset( 
            data_dir=conf['dataset']['data_dir'], 
            label_dir=conf['dataset']['label_dir'], 
            seq_len=conf['dataset']['seq_len'], 
            motions=conf['dataset']['motions'],
            stride=conf['dataset']['stride'],
            scaler=None
        )
        
        # save global scaler stats
        scaler_save_path = Path(conf['save_dir']) / "scaler.pt"
        scaler_save_path.parent.mkdir(parents=True, exist_ok=True)
        if not scaler_save_path.exists():
            print(f"💾 Saving scaler stats to {scaler_save_path}...")
            torch.save(dataset.get_scaler(), scaler_save_path)
        shuffle = True
        
    return DataLoader(
        dataset, 
        batch_size=conf['dataset']['batch_size'], 
        shuffle=shuffle, 
        num_workers=conf['dataset']['num_workers']
    )

def train(args):
    with open(args.config, 'r') as f:
        conf = yaml.safe_load(f)
    if args.resume:
        conf['train']['resume'] = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loader = get_dataloader(conf, split='train')
    
    model_name = conf['model']['name']
    if not hasattr(models, model_name):
        raise ValueError(f"models.py does not define class '{model_name}'")
    
    ModelClass = getattr(models, model_name)

    if model_name == 'TrajCVAE':
        model = ModelClass(
            traj_dim=conf['model']['traj_dim'], 
            cond_dim=conf['model']['cond_dim'],
            teacher_forcing=conf['model']['teacher_forcing_ratio'],
            past_lenth=conf['model']['past_lenth']
        ).to(device)
    print(f"Loaded model: {model_name}")

    optimizer = optim.Adam(model.parameters(), lr=conf['train']['lr'])

    ckpt_dir = Path(conf['save_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = ckpt_dir / f"{model_name}_last.pt"
    start_epoch = 0

    # Resume
    if conf['train']['resume'] and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optim"])
        start_epoch = ck["epoch"] + 1
        print(f"🔁 Resumed from epoch {start_epoch}")

    # Training loop
    best_loss = float("inf")
    criterion = nn.MSELoss()

    for epoch in range(start_epoch, conf['train']['epochs']):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        
        for i, batch in enumerate(loader):
            # x: (B, T, D_traj), y: (B, T, D_cond)
            x, y = batch[0].to(device), batch[1].to(device)
            
            optimizer.zero_grad()
            
            if model_name == 'TrajCVAE':
                pred, _, _ = model(x, y)
                loss = criterion(pred, x)
                recon_loss = loss.item()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % 20 == 0:
                print(f"Epoch {epoch} [{i}/{len(loader)}] Loss: {loss.item():.6f}")
                
        avg_loss = total_loss / len(loader)
        print(f"⏱️ Epoch {epoch} finished in {time.time()-t0:.1f}s | Avg Loss: {avg_loss:.6f}")

        state_dict = {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "epoch": epoch
        }
        torch.save(state_dict, last_ckpt)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_ckpt = ckpt_dir / f"{model_name}_best.pt"
            torch.save(state_dict, best_ckpt)
            print(f"🌟 New best model: {best_loss:.6f}")

        if (epoch + 1) % conf['train']['save_freq'] == 0:
            torch.save(state_dict, ckpt_dir / f"{model_name}_epoch_{epoch+1}.pt")

    print(f"🎉 Training complete! Checkpoints saved to: {ckpt_dir}")

if __name__ == "__main__":
    args = parse_cli()
    train(args)
