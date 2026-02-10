from pinocchio.robot_wrapper import RobotWrapper
import numpy as np

urdf_dir  = r"dataset\g1_retargeted_dataset\g1"
urdf_path = rf"{urdf_dir}\g1_29dof_rev_1_0.urdf"

robot = RobotWrapper.BuildFromURDF(
    urdf_path,
    package_dirs=[urdf_dir]   # so "meshes/..." resolves
)

print("nq:", robot.model.nq)
print("nv:", robot.model.nv)
print("njoints:", robot.model.njoints)
print("First joint names:", list(robot.model.names)[:15])
import numpy as np

names = list(robot.model.names)
qmin = robot.model.lowerPositionLimit
qmax = robot.model.upperPositionLimit

print("Index | Joint | qmin | qmax")
for j in range(1, robot.model.njoints):  # skip universe joint=0
    joint = robot.model.joints[j]
    if joint.nq == 1:
        idx = robot.model.idx_qs[j]
        print(f"{j:>3} | {names[j]:30s} | {qmin[idx]: .3f} | {qmax[idx]: .3f}")
