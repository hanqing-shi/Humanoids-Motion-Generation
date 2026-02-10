#!/usr/bin/env python3
import os, glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ----------------------------------
# Hyperparameters
# ----------------------------------
HORIZON = 30          # future frames (1 sec at 30Hz)
D_CONF  = 36          # humanoid config dims per frame (q_t)
COND_DIM = D_CONF + 3 # q_t (36) + [vx,vy,wz] (3) = 39
FUTURE_DIM = HORIZON * D_CONF  # 30 * 36 = 1080

BATCH_SIZE = 128
EPOCHS = 50
LR = 1e-3
LATENT_DIM = 32
BETA = 1.0            # weight on KL (β-VAE style)


# ----------------------------------
# Dataset
# ----------------------------------

class MotionDataset(Dataset):
    """
    Each sample:
      cond_t   = [ q_t (36), vx_t, vy_t, wz_t ]        -> shape (39,)
      future_q = [ q_{t+1} ... q_{t+30} ]              -> shape (30,36)
      future_q_flat -> shape (1080,)
    """
    def __init__(self, data_dir, horizon=HORIZON, stride=1, skip_header=True):
        self.conds = []     # (39,)
        self.futures = []   # (1080,)

        csv_list = sorted(glob.glob(os.path.join(data_dir, "*_vel.csv")))
        if not csv_list:
            raise RuntimeError(f"No *_vel.csv files found in {data_dir}")

        for path in csv_list:
            data = np.genfromtxt(path, delimiter=',', skip_header=1 if skip_header else 0)
            if data.ndim == 1:
                data = data.reshape(1, -1)

            if data.shape[1] != D_CONF + 3:
                raise RuntimeError(
                    f"{path}: expected {D_CONF+3} columns (36 conf + 3 vel), got {data.shape[1]}"
                )

            q_all   = data[:, :D_CONF]   # (T,36)
            vel_all = data[:, D_CONF:]   # (T,3)
            T = data.shape[0]

            for t in range(0, T - horizon - 1, stride):
                q_t      = q_all[t]                   # (36,)
                vel_t    = vel_all[t]                 # (3,)
                future_q = q_all[t+1:t+1+horizon]     # (H,36)

                cond_vec    = np.concatenate([q_t, vel_t], axis=0)          # (39,)
                future_flat = future_q.reshape(-1)                          # (H*36 =1080,)

                self.conds.append(cond_vec)
                self.futures.append(future_flat)

        self.conds   = torch.tensor(np.stack(self.conds), dtype=torch.float32)    # (N,39)
        self.futures = torch.tensor(np.stack(self.futures), dtype=torch.float32)  # (N,1080)

        print(f"[MotionDataset] N={len(self.conds)} samples, cond_dim={self.conds.shape[1]}, future_dim={self.futures.shape[1]}")

    def __len__(self):
        return self.conds.shape[0]

    def __getitem__(self, idx):
        return self.conds[idx], self.futures[idx]


# ----------------------------------
# CVAE Model
# ----------------------------------

class MotionCVAE(nn.Module):
    """
    Conditional VAE:
      Encoder: takes [cond, future_flat] -> mu, logvar
      Decoder: takes [cond, z] -> future_hat_flat
    """
    def __init__(self,
                 cond_dim=COND_DIM,
                 future_dim=FUTURE_DIM,
                 latent_dim=LATENT_DIM,
                 hidden_dim=512,
                 depth=4):
        super().__init__()
        self.cond_dim = cond_dim
        self.future_dim = future_dim
        self.latent_dim = latent_dim
        self.horizon = HORIZON
        self.d_conf = D_CONF

        # ----- Encoder -----
        enc_in_dim = cond_dim + future_dim  # concat(cond, future_flat)
        enc_layers = []
        dim = enc_in_dim
        for _ in range(depth):
            enc_layers.append(nn.Linear(dim, hidden_dim))
            enc_layers.append(nn.ReLU())
            dim = hidden_dim
        self.encoder_body = nn.Sequential(*enc_layers)
        self.to_mu     = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)

        # ----- Decoder -----
        dec_in_dim = cond_dim + latent_dim  # concat(cond, z)
        dec_layers = []
        dim = dec_in_dim
        for _ in range(depth):
            dec_layers.append(nn.Linear(dim, hidden_dim))
            dec_layers.append(nn.ReLU())
            dim = hidden_dim
        dec_layers.append(nn.Linear(hidden_dim, future_dim))
        self.decoder_body = nn.Sequential(*dec_layers)

    def encode(self, cond, future_flat):
        # cond: (B,39), future_flat: (B,1080)
        x = torch.cat([cond, future_flat], dim=-1)  # (B,1119)
        h = self.encoder_body(x)                    # (B,hidden)
        mu = self.to_mu(h)                          # (B,latent_dim)
        logvar = self.to_logvar(h)                  # (B,latent_dim)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, cond, z):
        # cond: (B,39), z: (B,latent_dim)
        d_in = torch.cat([cond, z], dim=-1)        # (B, 39+latent_dim)
        future_hat_flat = self.decoder_body(d_in)  # (B,1080)
        # reshape to (B,30,36) if you want sequence form
        future_hat_seq = future_hat_flat.view(-1, self.horizon, self.d_conf)
        return future_hat_flat, future_hat_seq

    def forward(self, cond, future_flat):
        mu, logvar = self.encode(cond, future_flat)
        z = self.reparameterize(mu, logvar)
        future_hat_flat, future_hat_seq = self.decode(cond, z)
        return future_hat_flat, future_hat_seq, mu, logvar


def cvae_loss(future_flat, future_hat_flat, mu, logvar, beta=BETA):
    # recon loss: MSE between true rollout and predicted rollout
    recon = nn.functional.mse_loss(future_hat_flat, future_flat, reduction='mean')
    # KL: push q(z|x,c) towards N(0,I)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon, kl


# ----------------------------------
# Training / Sampling
# ----------------------------------

def train_cvae(
    data_dir="retarget_out_vel",
    save_path="motion_cvae.pt",
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LR,
):

    dataset = MotionDataset(data_dir, horizon=HORIZON, stride=1, skip_header=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = MotionCVAE().train()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs+1):
        total_loss = 0.0
        total_rec  = 0.0
        total_kl   = 0.0
        total_n    = 0

        for cond, future_flat in loader:
            # cond:        (B,39)
            # future_flat: (B,1080)

            future_hat_flat, _, mu, logvar = model(cond, future_flat)
            loss, recon, kl = cvae_loss(future_flat, future_hat_flat, mu, logvar, beta=BETA)

            optim.zero_grad()
            loss.backward()
            optim.step()

            bs = cond.size(0)
            total_loss += loss.item()  * bs
            total_rec  += recon.item() * bs
            total_kl   += kl.item()    * bs
            total_n    += bs

        avg_loss = total_loss / total_n
        avg_rec  = total_rec  / total_n
        avg_kl   = total_kl   / total_n
        print(f"[{ep:03d}] loss={avg_loss:.6f}  recon={avg_rec:.6f}  kl={avg_kl:.6f}")

    torch.save({
        "model_state": model.state_dict(),
        "horizon": HORIZON,
        "d_conf": D_CONF,
        "latent_dim": LATENT_DIM,
        "beta": BETA,
    }, save_path)

    print(f"Saved CVAE model to {save_path}")


@torch.no_grad()
def sample_future(model, cond_vec, n_samples=1, device="cpu"):
    """
    cond_vec: np.array shape (39,)  = [q_t(36), vx,vy,wz]
    returns: list of n_samples trajectories,
             each is (30,36) numpy
    """
    model.eval()
    cond = torch.tensor(cond_vec, dtype=torch.float32, device=device).unsqueeze(0)  # (1,39)

    outputs = []
    for _ in range(n_samples):
        # sample z ~ N(0,I)
        z = torch.randn(1, model.latent_dim, device=device)
        future_hat_flat, future_hat_seq = model.decode(cond, z)
        # future_hat_seq: (1,30,36)
        outputs.append(future_hat_seq.squeeze(0).cpu().numpy())
    return outputs


if __name__ == "__main__":
    train_cvae(
        data_dir="retarget_out_vel",
        save_path="motion_cvae.pt",
        epochs=50,
        batch_size=128,
        lr=1e-3,
    )