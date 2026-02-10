import pinocchio as pin

robot = pin.RobotWrapper.BuildFromURDF(
    "./dataset/g1_retargeted_dataset/g1/g1_29dof_rev_1_0.urdf",
    "./dataset/g1_retargeted_dataset/g1",
    pin.JointModelFreeFlyer()
)

names = list(robot.model.names)

print("nq =", robot.model.nq, "nv =", robot.model.nv)
print("\nq indices 0..6 are the free-flyer base (not joints).")

# For each 1-DoF joint, print where it lives in q
for jid in range(1, robot.model.njoints):
    j = robot.model.joints[jid]
    if j.nq == 1:
        qi = robot.model.idx_qs[jid]
        print(f"q[{qi:2d}] -> {names[jid]}")
