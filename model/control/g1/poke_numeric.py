import numpy as np
import pinocchio as pin

robot = pin.RobotWrapper.BuildFromURDF(
    "./dataset/g1_retargeted_dataset/g1/g1_29dof_rev_1_0.urdf",
    "./dataset/g1_retargeted_dataset/g1",
    pin.JointModelFreeFlyer()
)
model, data = robot.model, robot.data
names = list(model.names)

q0 = pin.neutral(model)          # safe nominal
q1 = q0.copy()

k = 7  # <-- q index to poke (7..35 are joints)
q1[k] += 0.3

def joint_transforms(q):
    robot.framesForwardKinematics(q)
    # oMi: joint placements in world
    Ts = []
    for jid in range(model.njoints):
        T = robot.data.oMi[jid]
        Ts.append((T.translation.copy(), T.rotation.copy()))
    return Ts

T0 = joint_transforms(q0)
T1 = joint_transforms(q1)

# Find joints whose transforms changed the most
deltas = []
for jid in range(model.njoints):
    dp = np.linalg.norm(T1[jid][0] - T0[jid][0])
    dR = np.linalg.norm(T1[jid][1] - T0[jid][1])
    deltas.append((dp + dR, jid))

deltas.sort(reverse=True)
print(f"Poked q[{k}]")
print("Top moved joints:")
for score, jid in deltas[:10]:
    print(f"  {names[jid]:30s} score={score:.6f}")
