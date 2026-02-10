#!/usr/bin/env python3
"""
TrajCVAE motion generator wrapper that mirrors inference_rt.py.

- loads models.TrajCVAE
- loads checkpoint dict with keys like: {"model","optim","epoch"}
- maintains recurrent context:
    cond_past: (1, past_len, 3)
    x_past:    (1, past_len, 36)
    x_start:   (1, 1, 36)
- step(cond_future) returns samples: (T, 36) in *model format* (root quat = xyzw)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from model import models  # repo-root models.py


@dataclass
class TrajCVAESpec:
    traj_dim: int = 36
    cond_dim: int = 3
    past_length: int = 10
    teacher_forcing: float = 0.0


def _load_ckpt_state_dict(model: torch.nn.Module, ckpt_obj):
    """
    inference_rt checkpoint format: {'model': state_dict, 'optim': ..., 'epoch': ...}
    Also supports a few common alternatives.
    """
    if isinstance(ckpt_obj, dict):
        for k in ("model", "model_state_dict", "state_dict"):
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                sd = ckpt_obj[k]
                # strip DataParallel prefix if needed
                if any(key.startswith("module.") for key in sd.keys()):
                    sd = {key.replace("module.", "", 1): val for key, val in sd.items()}
                model.load_state_dict(sd)
                return

        # if it's already a raw state_dict
        if all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            sd = ckpt_obj
            if any(key.startswith("module.") for key in sd.keys()):
                sd = {key.replace("module.", "", 1): val for key, val in sd.items()}
            model.load_state_dict(sd)
            return

    raise RuntimeError(f"Unrecognized checkpoint format. Top-level keys: {list(ckpt_obj.keys()) if isinstance(ckpt_obj, dict) else type(ckpt_obj)}")


class TrajCVAEGenerator:
    def __init__(self, model: torch.nn.Module, device: str, spec: TrajCVAESpec, x_init_36_xyzw: np.ndarray):
        self.model = model
        self.device = device
        self.spec = spec
        self.model.eval()

        # initialize context exactly like inference_rt
        x_init = np.asarray(x_init_36_xyzw, dtype=np.float32).reshape(1, 1, spec.traj_dim)
        x_init_t = torch.from_numpy(x_init).to(device)

        self.cond_past = torch.zeros((1, spec.past_length, spec.cond_dim), device=device)
        self.x_past = x_init_t.repeat(1, spec.past_length, 1)   # (1, past, 36)
        self.x_start = x_init_t[:, -1:, :]                      # (1, 1, 36)

    @torch.no_grad()
    def sample(self, cond_future_np: np.ndarray) -> np.ndarray:
        """
        cond_future_np: (T,3) float
        returns: samples (T,36) float in model format (xyzw)
        """
        cond_future_np = np.asarray(cond_future_np, dtype=np.float32)
        if cond_future_np.ndim != 2 or cond_future_np.shape[1] != self.spec.cond_dim:
            raise ValueError(f"cond_future must be (T,3), got {cond_future_np.shape}")

        cond_future = torch.from_numpy(cond_future_np).to(self.device).unsqueeze(0)  # (1,T,3)

        samples = self.model.sample(self.cond_past, cond_future, self.x_past, self.x_start)  # (1,T,36)

        # update past buffers (mirror inference_rt)
        self.cond_past = torch.cat([self.cond_past, cond_future], dim=1)[:, -self.spec.past_length :, :]
        new_state = samples[:, -1:, :]  # (1,1,36)
        self.x_past = torch.cat([self.x_past, new_state], dim=1)[:, -self.spec.past_length :, :]
        self.x_start = new_state

        return samples.squeeze(0).detach().cpu().numpy().astype(np.float32)


def load_trajcvae_generator(
    ckpt_path: str,
    device: str = "cpu",
    past_length: int = 10,
    x_init_36_xyzw: Optional[np.ndarray] = None,
) -> TrajCVAEGenerator:
    """
    Loads models.TrajCVAE exactly like inference_rt.

    x_init_36_xyzw defaults to the same init vector in inference_rt.py
    (note: root quat is xyzw in that file / rerun convention).
    """
    if x_init_36_xyzw is None:
        x_init_36_xyzw = np.array([
            0.0, 0.0, 0.76,
            0.0, 0.0, 0.0, 1.0,  # root quat xyzw (rerun convention)
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
            0.0, 0.0, 0.0,
            0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
            0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0
        ], dtype=np.float32)

    spec = TrajCVAESpec(traj_dim=36, cond_dim=3, past_length=past_length, teacher_forcing=0.0)

    ModelClass = getattr(models, "TrajCVAE")
    model = ModelClass(
        traj_dim=spec.traj_dim,
        cond_dim=spec.cond_dim,
        teacher_forcing=spec.teacher_forcing,
        past_lenth=spec.past_length
    ).to(device)

    ckpt_file = Path(ckpt_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    ckpt = torch.load(str(ckpt_file), map_location=device)
    _load_ckpt_state_dict(model, ckpt)

    print(f"✅ Loaded TrajCVAE weights from: {ckpt_file}")
    return TrajCVAEGenerator(model=model, device=device, spec=spec, x_init_36_xyzw=x_init_36_xyzw)
