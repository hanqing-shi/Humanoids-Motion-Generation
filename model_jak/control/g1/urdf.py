import pinocchio as pin

URDF_DIR  = r"dataset\g1_retargeted_dataset\g1"
URDF_PATH = URDF_DIR + r"\g1_29dof_rev_1_0.urdf"

def load_robot():
    return pin.RobotWrapper.BuildFromURDF(
        URDF_PATH,
        URDF_DIR,
        pin.JointModelFreeFlyer()
    )
