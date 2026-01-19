
# Pinocchio joint indices (skip universe = 0)
LEFT_LEG  = list(range(1, 7))     # 1..6
RIGHT_LEG = list(range(7, 13))    # 7..12
WAIST     = list(range(13, 16))   # 13..15
LEFT_ARM  = list(range(16, 23))   # 16..22
RIGHT_ARM = list(range(23, 30))   # 23..29

ACTUATED_JOINT_IDS = list(range(1, 30)) # 1..29
ACTUATED_Q_IDXS = list(range(7,36))
N_ACTUATED = len(ACTUATED_Q_IDXS)
