import numpy as np
from model.control.g1.joints import ACTUATED_Q_IDXS, N_ACTUATED

FREE_FLYER_NQ = 7
FULL_NQ = FREE_FLYER_NQ + N_ACTUATED  # 36

def assert_full_configuration(q_full: np.ndarray, name: str = "q_full"):
    q_full = np.asarray(q_full)
    if q_full.shape != (FULL_NQ,):
        raise ValueError(f"{name} must have shape ({FULL_NQ},), got {q_full.shape}")

def assert_actuated(q_act: np.ndarray, name: str = "q_act"):
    q_act = np.asarray(q_act)
    if q_act.shape != (N_ACTUATED,):
        raise ValueError(f"{name} must have shape ({N_ACTUATED},), got {q_act.shape}")

def split_actuated(q_full: np.ndarray):
    assert_full_configuration(q_full)
    return q_full[ACTUATED_Q_IDXS]

def set_actuated(q_full: np.ndarray, q_act: np.ndarray):
    assert_full_configuration(q_full)
    assert_actuated(q_act)
    q_out = q_full.copy()
    q_out[ACTUATED_Q_IDXS] = q_act
    return q_out
