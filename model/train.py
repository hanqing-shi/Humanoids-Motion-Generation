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
import math

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

def split_root_joints(x):
    """
    x: (..., 36) with ordering:
       [root_x, root_y, root_z,
        root_qx, root_qy, root_qz, root_qw,
        joint_0, ..., joint_28]
    Returns:
        pos:  (..., 3)
        quat: (..., 4) as [qx, qy, qz, qw]
        joints: (..., 29)
    """
    pos = x[..., 0:3]
    quat = x[..., 3:7]
    joints = x[..., 7:]
    return pos, quat, joints

def finite_diff(x, dt):
    """
    Finite difference along time dimension: x: (B, T, D) -> (B, T-1, D)
    """
    return (x[:, 1:] - x[:, :-1]) / dt

def normalize_quat(q, eps=1e-8):
    """
    Normalize quaternions q: (..., 4) in [qx, qy, qz, qw] order.
    """
    norm = torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(eps)
    return q / norm

def quat_geodesic_loss(q_pred, q_true):
    """
    Geodesic distance between unit quaternions, averaged over batch+time.
    Both q_pred and q_true are (..., 4) in [qx, qy, qz, qw] order.
    """
    q_pred_n = normalize_quat(q_pred)
    q_true_n = normalize_quat(q_true)
    # convert to [w, x, y, z] for easier dot math
    qw_p = q_pred_n[..., 3]
    qx_p = q_pred_n[..., 0]
    qy_p = q_pred_n[..., 1]
    qz_p = q_pred_n[..., 2]

    qw_t = q_true_n[..., 3]
    qx_t = q_true_n[..., 0]
    qy_t = q_true_n[..., 1]
    qz_t = q_true_n[..., 2]

    dot = qw_p*qw_t + qx_p*qx_t + qy_p*qy_t + qz_p*qz_t
    dot = torch.clamp(dot.abs(), -1.0, 1.0)
    angles = 2.0 * torch.arccos(dot)
    return (angles ** 2).mean()

def quat_to_matrix(q):
    """
    Convert quaternions [qx,qy,qz,qw] to rotation matrices (...,3,3).
    """
    qn = normalize_quat(q)
    qx, qy, qz, qw = qn.unbind(dim=-1)
    # from standard formula
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    ww = qw * qw
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    m00 = ww + xx - yy - zz
    m01 = 2 * (xy - wz)
    m02 = 2 * (xz + wy)

    m10 = 2 * (xy + wz)
    m11 = ww - xx + yy - zz
    m12 = 2 * (yz - wx)

    m20 = 2 * (xz - wy)
    m21 = 2 * (yz + wx)
    m22 = ww - xx - yy + zz

    row0 = torch.stack([m00, m01, m02], dim=-1)
    row1 = torch.stack([m10, m11, m12], dim=-1)
    row2 = torch.stack([m20, m21, m22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)

def quat_to_yaw(q):
    """
    Extract yaw (rotation about z) from quats [qx,qy,qz,qw] -> (...,).
    """
    qn = normalize_quat(q)
    qx, qy, qz, qw = qn.unbind(dim=-1)
    # ZYX yaw from unit quaternion
    sin_yaw = 2 * (qw * qz + qx * qy)
    cos_yaw = 1 - 2 * (qy * qy + qz * qz)
    yaw = torch.atan2(sin_yaw, cos_yaw)
    return yaw

def wrap_angle(a):
    """
    Wrap angle to [-pi, pi].
    """
    return torch.remainder(a + math.pi, 2*math.pi) - math.pi

def kl_divergence(mu, logvar):
    """KL( q(z|.) || N(0,I) ) per-sample"""
    return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=-1)

def get_dataloader(batch_size, dataset, num_workers=0, shuffle=True):
    if dataset == 'feature':
        dataset = MotionDataset( 
                    data_dir = './dataset/data_feature', 
                    label_dir = './dataset/data_label', 
                    seq_len=30, # choose length of each sequence
                    motions = ['walk'],
                    columns=("pos", "ori"),
                    stride=10,
                    transform=None)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    if dataset == 'joint':
        dataset = JointDataset( 
                    data_dir = './dataset/data_joint', 
                    label_dir = './dataset/data_label', 
                    seq_len=30, 
                    motions = ['walk'],
                    stride=10,
                    transform=None)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

def train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Initialize dataloader
    loader = get_dataloader(args.batch_size, args.dataset, num_workers=0)
    
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
                # concatenate current pose and future velocity label
                c = torch.cat([x[:, 0, :].flatten(1), y.flatten(1)], dim=-1)

            optimizer.zero_grad()

            # Forward pass
            pred, mu, logvar = model(x, c)

            kl_loss = kl_divergence(mu, logvar).mean()
            beta = 1.0

            if args.dataset == "joint":
                # --- Joint-space training with root- and command-aware losses ---
                dt = 0.03  # must match extract_features_labels.py

                pos_true, quat_true, joints_true = split_root_joints(x)
                pos_pred, quat_pred, joints_pred = split_root_joints(pred)

                # Root position loss
                L_pos = F.mse_loss(pos_pred, pos_true)

                # Root world linear velocity (finite diff)
                v_world_true = finite_diff(pos_true, dt)  # (B,T-1,3)
                v_world_pred = finite_diff(pos_pred, dt)
                L_vel = F.mse_loss(v_world_pred, v_world_true)

                # Root orientation geodesic loss
                L_ori = quat_geodesic_loss(quat_pred, quat_true)

                # Command-consistency loss: match label velocities (local frame)
                # Labels y are (B, T, 3): [linear_x_local, linear_y_local, angular_z_local]
                R_world_from_base = quat_to_matrix(quat_pred)          # (B,T,3,3)
                R_base_from_world = R_world_from_base.transpose(-1, -2)

                # Align indices so velocities at step t use orientation at t
                R_base_from_world_step = R_base_from_world[:, 1:, :, :]  # (B,T-1,3,3)

                # Transform world velocities into base frame
                v_local_pred = torch.einsum("btij,btj->bti", R_base_from_world_step, v_world_pred)

                # Yaw rate from quaternions
                yaw_pred = quat_to_yaw(quat_pred)             # (B,T)
                delta_yaw = wrap_angle(yaw_pred[:, 1:] - yaw_pred[:, :-1])
                omega_z_pred = delta_yaw / dt                 # (B,T-1)

                cmd_pred = torch.stack(
                    [v_local_pred[..., 0], v_local_pred[..., 1], omega_z_pred],
                    dim=-1,
                )  # (B,T-1,3)

                cmd_true = y[:, 1:, :]  # drop first label to match T-1
                L_cmd = F.mse_loss(cmd_pred, cmd_true)

                # Joint-angle reconstruction loss
                L_joints = F.mse_loss(joints_pred, joints_true)

                # Weighting of different terms
                w_pos = 1.0
                w_vel = 0.2
                w_ori = 0.5
                w_cmd = 1.0
                w_joints = 0.1

                loss = (
                    w_pos * L_pos
                    + w_vel * L_vel
                    + w_ori * L_ori
                    + w_cmd * L_cmd
                    + w_joints * L_joints
                    + beta * kl_loss
                )

                # for logging
                recon_loss = torch.tensor(0.0, device=x.device)
            else:
                # --- Original feature-space reconstruction loss ---
                recon_loss = F.mse_loss(x, pred, reduction="none").sum(dim=(1, 2)).mean()
                loss = recon_loss + beta * kl_loss

            # Backpropagation
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % log_interval == 0:
                print(
                    f"Epoch {epoch} [{i}/{len(loader)}] "
                    f"Loss: {loss.item():.6f} | Recon: {recon_loss.item():.6f} | KL: {kl_loss.item():.6f}"
                )

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
