import torch
from model.models import TrajCVAE

ckpt = torch.load("checkpoints/TrajCVAE_best_walk.pt", map_location="cpu")["model"]

m = TrajCVAE(
    traj_dim=36, cond_dim=3,
    teacher_forcing=0, past_lenth=10,
    fut_hidden=256, z_dim=64,
    dec_hidden=256, num_layers=1
)

m.load_state_dict(ckpt, strict=True)
print("✅ Checkpoint matches TrajCVAE config exactly.")
