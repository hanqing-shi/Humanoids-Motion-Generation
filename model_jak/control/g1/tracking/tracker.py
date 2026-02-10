# Hybrid PD tracker (impedance-style) that:
# - smooths the reference
# - enforces per-joint velocity limits (and optional accel limits)
# - outputs (q_cmd, dq_cmd, kp, kd, tau_ff)
#
# Designed to work in sim now and plug into hardware later.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class HybridPDParams:
    alpha: float = 0.25 # Reference smoothing: q_ref <- (1-alpha)*q_ref + alpha*q_des - increase if sim isnt responsive enough
    ###### Limits (applied to command generation)
    dq_limit: float = 4.0                   # rad/s (per-joint default)
    ddq_limit: Optional[float] = None       # rad/s^2

    # PD gains
    kp_default: float = 60.0                # Stiffness
    kd_default: float = 6.0                 # Damping
    
    use_critical_damping: bool = True       # If True, set kd ~= 2*sqrt(kp) (critical-ish damping) per joint
    output_dq_cmd: bool = True              # Whether to output dq_cmd as "feedforward velocity" (finite-diff of q_cmd)

class HybridPDTracker:
    """
    Mode-agnostic joint-space tracker.
    Inputs: q_des (desired joints), q_meas/dq_meas (measured joints)
    Outputs: q_cmd, dq_cmd, kp, kd, tau_ff

    q_cmd is filtered/limited.
    dq_cmd is either finite-diff(q_cmd) or zeros depending on params.output_dq_cmd.
    """
    def __init__(
        self,
        n: int,
        dt: float,
        params: HybridPDParams = HybridPDParams(),
        dq_limit: Optional[np.ndarray] = None,
        ddq_limit: Optional[np.ndarray] = None,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
    ):
        self.n = int(n)
        self.dt = float(dt)
        self.p = params

        self.q_ref: Optional[np.ndarray] = None
        self.q_cmd_prev: Optional[np.ndarray] = None
        self.dq_cmd_prev: Optional[np.ndarray] = None

        # Per-joint limits / gains (optional)
        self.dq_limit = self._make_vec(dq_limit, self.p.dq_limit)
        self.ddq_limit = None
        if self.p.ddq_limit is not None or ddq_limit is not None:
            default = self.p.ddq_limit if self.p.ddq_limit is not None else 0.0
            self.ddq_limit = self._make_vec(ddq_limit, default)

        self.kp = self._make_vec(kp, self.p.kp_default)

        if kd is None:
            if self.p.use_critical_damping:
                # kd ~= 2 * sqrt(kp) (units consistent if position error is rad)
                self.kd = 2.0 * np.sqrt(self.kp)
            else:
                self.kd = self._make_vec(None, self.p.kd_default)
        else:
            self.kd = self._make_vec(kd, self.p.kd_default)

    def _make_vec(self, arr: Optional[np.ndarray], default: float) -> np.ndarray:
        if arr is None:
            return np.full((self.n,), float(default), dtype=np.float32)
        v = np.asarray(arr, dtype=np.float32).reshape(-1)
        if v.shape[0] != self.n:
            raise ValueError(f"Expected vector of length {self.n}, got {v.shape}")
        return v

    def reset(self, q0: np.ndarray):
        q0 = np.asarray(q0, dtype=np.float32).reshape(-1)
        if q0.shape[0] != self.n:
            raise ValueError(f"reset expects ({self.n},), got {q0.shape}")
        self.q_ref = q0.copy()
        self.q_cmd_prev = q0.copy()
        self.dq_cmd_prev = np.zeros((self.n,), dtype=np.float32)

    def step(
        self,
        q_des: np.ndarray,
        q_meas: np.ndarray,
        dq_meas: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns: (q_cmd, dq_cmd, kp, kd, tau_ff)
        """
        q_des = np.asarray(q_des, dtype=np.float32).reshape(-1)
        q_meas = np.asarray(q_meas, dtype=np.float32).reshape(-1)
        dq_meas = np.asarray(dq_meas, dtype=np.float32).reshape(-1)
        if q_des.shape[0] != self.n or q_meas.shape[0] != self.n or dq_meas.shape[0] != self.n:
            raise ValueError("q_des, q_meas, dq_meas must all be length n")

        if self.q_ref is None:
            self.reset(q_meas)

        # Smooths reference with alpha
        alpha = float(self.p.alpha)
        self.q_ref = (1.0 - alpha) * self.q_ref + alpha * q_des

        # 2) Convert reference into a command with velocity (and optional accel) limiting
        # Desired delta this timestep
        dq_des_cmd = (self.q_ref - self.q_cmd_prev) / self.dt  # "commanded velocity" implied by new ref
        dq_limited = np.clip(dq_des_cmd, -self.dq_limit, self.dq_limit) # velocity limit
        if self.ddq_limit is not None and self.dq_cmd_prev is not None: # accel limit
            ddq = (dq_limited - self.dq_cmd_prev) / self.dt
            ddq_limited = np.clip(ddq, -self.ddq_limit, self.ddq_limit)
            dq_limited = self.dq_cmd_prev + ddq_limited * self.dt

        q_cmd = self.q_cmd_prev + dq_limited * self.dt
        if self.p.output_dq_cmd:
            dq_cmd = dq_limited
        else:
            dq_cmd = np.zeros((self.n,), dtype=np.float32)

        tau_ff = np.zeros((self.n,), dtype=np.float32)  # feedforward torque
        # update state
        self.q_cmd_prev = q_cmd.copy()
        self.dq_cmd_prev = dq_cmd.copy()

        return q_cmd, dq_cmd, self.kp.copy(), self.kd.copy(), tau_ff
