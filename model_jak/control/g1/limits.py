import numpy as np
from .urdf import load_robot
from .joints import N_ACTUATED

def get_joint_limits():
    """
    Returns limits for the 29 actuated joints in q[7:36] order,
    matching your runtime configuration vector (36-D).
    """
    robot = load_robot()
    m = robot.model
    names = list(m.names)

    qmin_full = m.lowerPositionLimit.copy()
    qmax_full = m.upperPositionLimit.copy()

    # limits for actuated joints (q[7]..q[35])
    qmin = qmin_full[7:36]
    qmax = qmax_full[7:36]

    # map q-index -> joint name using idx_qs
    qidx_to_name = {}
    for jid in range(m.njoints):
        if m.joints[jid].nq == 1:
            qi = m.idx_qs[jid]
            qidx_to_name[qi] = names[jid]

    joint_names = [qidx_to_name[i] for i in range(7, 36)]

    return qmin, qmax, joint_names


def save_joint_limits(prefix="model/control/g1"):
    qmin, qmax, names = get_joint_limits()
    np.save(f"{prefix}/qmin.npy", qmin)
    np.save(f"{prefix}/qmax.npy", qmax)
    np.save(f"{prefix}/joint_names.npy", names)
