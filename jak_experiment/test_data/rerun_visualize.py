import csv
from scipy.spatial.transform import Rotation as R
import numpy as np
import pinocchio as pin

def compute_dq(q_prev: np.ndarray, q_curr: np.ndarray, dt: float) -> np.ndarray:
    """
    Compute the generalized velocity dq from two configurations (q_prev, q_curr)
    for a Pinocchio model with a free-flyer base.

    The first 7 DoF of q correspond to:
        [x, y, z, qx, qy, qz, qw]
    and dq has 6 DoF for the base (3 linear + 3 angular) 
    plus joint velocities for the rest.

    Args:
        q_prev (np.ndarray): previous configuration [nq]
        q_curr (np.ndarray): current configuration [nq]
        dt (float): timestep (seconds)

    Returns:
        np.ndarray: dq [nv = 6 + (nq - 7)]
    """
    # --- derive sizes ---
    n_joints = len(q_curr) - 7          # number of internal joints
    dq = np.zeros(6 + n_joints)         # base(6) + joint(n)
    #print("n_joints:", n_joints)
    # --- 1. base linear velocity ---
    dq[0:3] = (q_curr[0:3] - q_prev[0:3]) / dt

    # --- 2. base angular velocity (from quaternion delta) ---
    quat_prev = q_prev[3:7]  # [x, y, z, w]
    quat_curr = q_curr[3:7]

    R_prev = R.from_quat(quat_prev)
    R_curr = R.from_quat(quat_curr)
    R_rel = R_prev.inv() * R_curr
    rotvec = R_rel.as_rotvec()  # rotation vector (rad)
    dq[3:6] = rotvec / dt       # angular velocity in world frame

    # --- 3. joint velocities ---
    dq[6:] = (q_curr[7:] - q_prev[7:]) / dt

    return dq



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, help="File name", default='walk1_subject1')
    parser.add_argument('--robot_type', type=str, help="Robot type", default='g1')
    args = parser.parse_args()

    file_name = args.file_name
    robot_type = args.robot_type
    csv_files = './' + file_name + '.csv'
    data = np.genfromtxt(csv_files, delimiter=',')

    robot = pin.RobotWrapper.BuildFromURDF('./g1_29dof_rev_1_0.urdf', './g1', pin.JointModelFreeFlyer())
    data_out = []

    dt = 0.03
    frame_names = []
    # 所有frame names
    for visual in robot.visual_model.geometryObjects:
            frame_names.append(visual.name[:-2])
   
    header = ['root_x', 'root_y', 'root_z',
              'root_qw', 'root_qx', 'root_qy', 'root_qz',
              'root_vx', 'root_vy', 'root_vz',
              'root_wx', 'root_wy', 'root_wz']
    for f in frame_names:
        header += [
            f'{f}_pos_x', f'{f}_pos_y', f'{f}_pos_z',
            f'{f}_ori_w', f'{f}_ori_x', f'{f}_ori_y', f'{f}_ori_z',
            f'{f}_vel_x', f'{f}_vel_y', f'{f}_vel_z',
            f'{f}_angvel_x', f'{f}_angvel_y', f'{f}_angvel_z'
        ]

    prev_conf = data[0]
    for frame_nr in range(data.shape[0]):

        configuration = data[frame_nr, :]
        
        # 用差分估算速度
        if frame_nr > 0:
            dq = compute_dq(prev_conf, configuration, dt)
        else:
            dq = np.zeros(robot.model.nv)
        dq = np.asarray(dq, dtype=np.float64)
        #print(configuration.shape, configuration.dtype)
        #print(dq.shape, dq.dtype)

        pin.forwardKinematics(robot.model, robot.data, configuration, dq)
        pin.updateFramePlacements(robot.model, robot.data)

        # root pose/orientation
        root_tf = robot.data.oMi[1]
        pos_root_world = root_tf.translation
        rot_root_world = R.from_matrix(root_tf.rotation).as_quat()
        rot_root_world = np.roll(rot_root_world, 1)  # xyzw -> wxyz

        prev_conf = configuration
         # root velocity
        v_root_world = dq[:3]   # linear vel in world
        w_root_world = dq[3:6]  # angular vel in world

        # root linear and angular velocity
        row = list(pos_root_world) + list(rot_root_world) + list(v_root_world) + list(w_root_world)

        ### base_tf.inverse() * frame_tf
        for frame_name in frame_names:
            frame_id = robot.model.getFrameId(frame_name)
            tf = robot.data.oMf[frame_id]
            pos_body_world = tf.translation
            #rot_body_world = R.from_matrix(tf.rotation).as_quat()
            #rot_body_world = np.roll(rot_body_world, 1) # xyzw -> wxyz

            # get velocity in WORLD
            v = pin.getFrameVelocity(robot.model, robot.data, frame_id, pin.ReferenceFrame.WORLD)
            v_linear = v.linear
            v_angular = v.angular

            T_world_from_base = robot.data.oMi[1]
            R_world_from_base = robot.data.oMi[1].rotation
            T_base_from_world = T_world_from_base.inverse()
            R_base_from_world = R_world_from_base.T

            pos_body_local = T_base_from_world.act(pos_body_world)

            R_world_from_body = tf.rotation                          # body 在世界下的旋转矩阵
            R_base_from_body = R_base_from_world @ R_world_from_body # body 在 base 下的旋转矩阵
            rot_body_local = R.from_matrix(R_base_from_body).as_quat()
            rot_body_local = np.roll(rot_body_local, 1)              # xyzw -> wxyz

            v_linear_local = R_base_from_world @ v.linear
            v_angular_local = R_base_from_world @ v.angular

            row += list(pos_body_local) + list(rot_body_local) + list(v_linear_local) + list(v_angular_local)
        data_out.append(row)

    out_name = robot_type + '/' + file_name + '_body.csv'
    with open(out_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data_out)

    print(f"✅ Generated {out_name} with {len(data_out)} frames and {len(header)} columns")
