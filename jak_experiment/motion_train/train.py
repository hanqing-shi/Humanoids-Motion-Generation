# train.py
import os
import argparse
import torch
from torch.utils.data import DataLoader
from dataset import MotionStateDataset
from models import TrajCVAE

# -------------------
# Quaternion utilities
# -------------------
def normalize_quat(q):
    return q / torch.clamp(q.norm(dim=-1, keepdim=True), min=1e-8)

def quat_geodesic_loss_safe(q1, q2, eps=1e-6):
    q1 = normalize_quat(q1); q2 = normalize_quat(q2)
    dot = torch.sum(q1 * q2, dim=-1)
    dot = torch.clamp(dot, -1.0 + eps, 1.0 - eps)
    return 2.0 * torch.arccos(torch.abs(dot))

# -------------------
# Collate
# -------------------
def collate_optional_cond_and_body(batch):
    # batch: list of (cond or None, target[T,D], body[T,3B] or None, frame_index)
    conds, targets, bodies, frames = zip(*batch)
    targets = torch.stack(targets, 0)
    frames = torch.tensor(frames, dtype=torch.long)
    conds_t = None if conds[0] is None else torch.stack(conds, 0)
    bodies_t = None if bodies[0] is None else torch.stack(bodies, 0)
    return conds_t, targets, bodies_t, frames

# -------------------
# Train one epoch
# -------------------
def train_one_epoch(
    model, loader, opt, device, quat_slice,
    max_steps=None, kl_scale: float = 1.0,
    lam_root: float = 1e-3, lam_joint: float = 1e-3, lam_body: float = 1e-1,
    log_every: int = 100,
):
    model.train()
    steps = 0
    agg = {"loss": 0.0, "pos": 0.0, "quat": 0.0, "joints": 0.0, "kl": 0.0,
           "root_s": 0.0, "joint_s": 0.0, "body": 0.0}

    def d1(x):  # finite difference in time
        return x[:, 1:] - x[:, :-1]

    for cond, target, body, _ in loader:
        if cond is not None:
            cond = cond.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        body = body.to(device, non_blocking=True) if body is not None else None

        # forward
        yhat_core, yhat_body, mu, logv = model(target, cond)

        # losses (core)
        pos_loss = torch.mean((yhat_core[..., :3] - target[..., :3]) ** 2)

        q_pred = yhat_core[..., quat_slice].contiguous()
        q_true = target[..., quat_slice].contiguous()
        quat_loss = torch.mean(quat_geodesic_loss_safe(q_pred, q_true))

        j_start = 13
        joints_loss = torch.mean((yhat_core[..., j_start:] - target[..., j_start:]) ** 2)

        # KL stabilized
        logv_safe = logv.clamp(min=-10.0, max=10.0)
        kl_loss = 0.5 * torch.mean(torch.exp(logv_safe) + mu ** 2 - 1.0 - logv_safe)

        # temporal smoothness (tiny)
        root_smooth = (d1(yhat_core[..., :13]) ** 2).mean()
        joints_smooth = (d1(yhat_core[..., 13:]) ** 2).mean()

        # aux body loss (positions only)
        body_loss = 0.0
        if yhat_body is not None and body is not None:
            body_loss = torch.mean((yhat_body - body) ** 2)

        loss = (
            pos_loss
            + 0.5 * quat_loss
            + joints_loss
            + (1e-3 * kl_scale) * kl_loss
            + lam_root * root_smooth
            + lam_joint * joints_smooth
            + lam_body * body_loss
        )

        if not torch.isfinite(loss):
            print("[warn] non-finite loss; skipping batch")
            opt.zero_grad(set_to_none=True)
            continue

        # step
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        # progress heartbeat
        if (steps % log_every) == 0:
            # defensively convert tensors to floats
            pl = float(pos_loss.detach())
            ql = float(quat_loss.detach())
            jl = float(joints_loss.detach())
            kl = float(kl_loss.detach())
            bl = float(body_loss if not isinstance(body_loss, float) else 0.0)
            print(f"  step {steps:05d} | loss {float(loss.detach()):.4f} "
                f"| pos {pl:.4f} | quat {ql:.4f} | joints {jl:.4f} | kl {kl:.4f} | body {bl:.4f}",
                flush=True)

        # log
        steps += 1
        agg["loss"] += float(loss.detach())
        agg["pos"] += float(pos_loss.detach())
        agg["quat"] += float(quat_loss.detach())
        agg["joints"] += float(joints_loss.detach())
        agg["kl"] += float(kl_loss.detach())
        agg["root_s"] += float(root_smooth.detach())
        agg["joint_s"] += float(joints_smooth.detach())
        agg["body"] += float(body_loss if isinstance(body_loss, float) else body_loss.detach())

        if max_steps and steps >= max_steps:
            break

    for k in agg:
        if steps > 0:
            agg[k] /= steps
    return agg

# -------------------
# Export viewer-ready CSV
# -------------------
@torch.no_grad()
def export_rollout_csv(model, loader, device, out_csv, quat_slice):
    import pandas as pd
    model.eval()
    cond, target, body, _ = next(iter(loader))
    if cond is not None:
        cond = cond.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    y_core, _, _, _ = model(target, cond)
    y_core = y_core.cpu()

    # renormalize quats in output
    q = normalize_quat(y_core[..., quat_slice])
    y_core = torch.cat([y_core[..., :quat_slice.start], q, y_core[..., quat_slice.stop:]], dim=-1)

    y = y_core[0].numpy()  # (T, D)
    D = y.shape[1]
    j_start = 13
    cols = (
        ["root_x", "root_y", "root_z",
         "root_w", "root_qx", "root_qy", "root_qz",
         "root_vx", "root_vy", "root_vz",
         "root_omx", "root_omy", "root_omz"]
        + [f"joint_{i+1}" for i in range(D - j_start)]
    )
    import pandas as pd
    df = pd.DataFrame(y, columns=cols)
    df.insert(0, "frame", range(len(df)))
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"🟢 Wrote rollout CSV → {out_csv}")

# -------------------
# Main
# -------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seq_len", type=int, default=120)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--use_cond", action="store_true")
    ap.add_argument("--use_body_aux", action="store_true")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=None)
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds = MotionStateDataset(
        args.data_dir, "train", args.seq_len, args.stride, args.use_cond, args.use_body_aux
    )
    val_ds = MotionStateDataset(
        args.data_dir, "val", args.seq_len, args.stride, args.use_cond, args.use_body_aux,
        stats_path=os.path.join(args.data_dir, "norm_stats.json"),
        body_stats_path=os.path.join(args.data_dir, "body_norm_stats.json")
    )

    # sample for dims
    sample = val_ds[0]
    cond_dim = 0 if sample[0] is None else sample[0].shape[-1]
    target_dim = sample[1].shape[-1]
    aux_dim = 0 if sample[2] is None else sample[2].shape[-1]  # 3*Nbodies
    quat_slice = slice(3, 7)  # (w, x, y, z)

    model = TrajCVAE(
        target_dim=target_dim,
        cond_dim=cond_dim,
        hidden=256,
        latent=64,
        num_layers=2,
        dropout=0.1,
        aux_dim=aux_dim,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    collate = collate_optional_cond_and_body
    train_ld = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate
    )
    val_ld = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate
    )

    for ep in range(1, args.epochs + 1):
        # KL warm-up: 0 -> 1 over first 10 epochs
        kl_scale = min(1.0, ep / 10.0)
        if ep <= 10:
            print(f"[epoch {ep}] KL scale = {kl_scale:.3f}")

        stats = train_one_epoch(
            model, train_ld, opt, device, quat_slice,
            max_steps=args.max_steps, kl_scale=kl_scale,
            log_every=args.log_every
        )

        print(f"Epoch {ep:03d} | loss {stats['loss']:.4f} | pos {stats['pos']:.4f} "
              f"| quat {stats['quat']:.4f} | joints {stats['joints']:.4f} "
              f"| kl {stats['kl']:.4f} | body {stats['body']:.4f}")

        # checkpoint + rollout CSV
        ckpt = os.path.join(args.out_dir, f"ckpt_epoch{ep:03d}.pt")
        torch.save({"model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "args": vars(args)}, ckpt)
        export_rollout_csv(
            model, val_ld, device,
            os.path.join(args.out_dir, f"rollout_eval_epoch{ep:03d}.csv"),
            quat_slice
        )

    final = os.path.join(args.out_dir, "model_final.pt")
    torch.save({"model": model.state_dict(),
                "opt": opt.state_dict(),
                "args": vars(args)}, final)
    print("💾 Saved final model", final)

if __name__ == "__main__":
    main()
