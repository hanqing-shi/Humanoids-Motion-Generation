import numpy as np
from .limits import get_joint_limits
from .joints import N_ACTUATED

class G1SafetyFilter:
    def __init__(
        self,
        dt,
        limit_margin_rad=np.deg2rad(5.0),
        dq_max_rad_s=2.0,
    ):
        """
        dt: control timestep (seconds)
        limit_margin_rad: keep away from hard joint stops
        dq_max_rad_s: max allowed joint velocity
        """
        self.dt = dt
        self.limit_margin = limit_margin_rad
        self.dq_max = dq_max_rad_s

        # Load limits once
        self.qmin, self.qmax, self.joint_names = get_joint_limits()
        self.prev_q = None

    def reset(self, q_init):
        """Initialize filter state (call once at startup)."""
        self.prev_q = np.asarray(q_init).copy()

    def step(self, q_cmd):
        """
        Apply safety filtering to commanded joints.

        q_cmd: (29,) array-like joint command
        returns: (q_safe, info_dict)
        """
        q_cmd = np.asarray(q_cmd).reshape(-1)

        info = {"ok": True, "reason": None}

        if q_cmd.shape[0] != N_ACTUATED:
            info["ok"] = False
            info["reason"] = "wrong_dim"
            return self.prev_q, info

        if not np.all(np.isfinite(q_cmd)):
            info["ok"] = False
            info["reason"] = "nan_or_inf"
            return self.prev_q, info

        # clip q to joint limits
        q_lo = self.qmin + self.limit_margin
        q_hi = self.qmax - self.limit_margin
        q_safe = np.clip(q_cmd, q_lo, q_hi)

        # rate limits
        if self.prev_q is not None:
            max_step = self.dq_max * self.dt
            dq = q_safe - self.prev_q
            dq = np.clip(dq, -max_step, max_step)
            q_safe = self.prev_q + dq

        # Update internal state
        self.prev_q = q_safe.copy()

        return q_safe, info
