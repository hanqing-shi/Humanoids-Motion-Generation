import csv
from scipy.spatial.transform import Rotation as R
import numpy as np
import pinocchio as pin
import os
import glob
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


def compute_dq(q_prev: np.ndarray, q_curr: np.ndarray, dt: float) -> np.ndarray:
    n_joints = len(q_curr) - 7
    dq = np.zeros(6 + n_joints)
    dq[0:3] = (q_curr[0:3] - q_prev[0:3]) / dt

    quat_prev = q_prev[3:7]
    quat_curr = q_curr[3:7]
    R_prev = R.from_quat(quat_prev)
    R_curr = R.from_quat(quat_curr)
    R_rel = R_prev.inv() * R_curr
    dq[3:6] = R_rel.as_rotvec() / dt
    dq[6:] = (q_curr[7:] - q_prev[7:]) / dt
    return dq

def butter_lowpass_filtfilt(x, dt, cutoff_hz=3.0, order=3):
    nyq = 0.5 / dt
    Wn = cutoff_hz / nyq
    b, a = butter(order, Wn, btype="low", analog=False)
    return filtfilt(b, a, x, axis=0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='./g1_clipped_retargeted_dataset/walk_test/', help="Input folder containing CSV files")
    parser.add_argument('--feature_out_folder', type=str, default='./data_feature/walk', help="Output folder for processed files")
    parser.add_argument('--label_out_folder', type=str, default='./data_test_label/walk', help="Output folder for label files")
    parser.add_argument('--robot_type', type=str, default='g1', help="Robot type (URDF folder name)")
    parser.add_argument('--plot', action='store_true', help="Save velocity-vs-time plots next to outputs")
    args = parser.parse_args()

    folder = args.folder
    feature_out_folder = args.feature_out_folder
    label_out_folder = args.label_out_folder
    robot_type = args.robot_type
    dt = 0.03

    os.makedirs(feature_out_folder, exist_ok=True)
    os.makedirs(label_out_folder, exist_ok=True)

    robot = pin.RobotWrapper.BuildFromURDF(
        f'./g1_clipped_retargeted_dataset/{robot_type}/{robot_type}_29dof_rev_1_0.urdf', f'./g1_clipped_retargeted_dataset/{robot_type}', pin.JointModelFreeFlyer()
    )

    frame_names = [visual.name[:-2] for visual in robot.visual_model.geometryObjects]
    feature_header = ['root_x', 'root_y', 'root_z',
              'root_qw', 'root_qx', 'root_qy', 'root_qz',
              'root_vx', 'root_vy', 'root_vz',
              'root_wx', 'root_wy', 'root_wz']
    for f in frame_names:
        feature_header += [
            f'{f}_pos_x', f'{f}_pos_y', f'{f}_pos_z',
            f'{f}_ori_w', f'{f}_ori_x', f'{f}_ori_y', f'{f}_ori_z',
            f'{f}_vel_x', f'{f}_vel_y', f'{f}_vel_z',
            f'{f}_angvel_x', f'{f}_angvel_y', f'{f}_angvel_z'
        ]

    label_header = ['linear_x', 'linear_y', 'angular_z']
    csv_files = sorted(glob.glob(os.path.join(folder, '*.csv')))
    print(f"📂 Found {len(csv_files)} CSV files in {folder}")

    for csv_path in csv_files:
        file_name = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"Processing {file_name} ...")

        data = np.genfromtxt(csv_path, delimiter=',')
        feature_data_out = []
        label_data_out = []
        prev_conf = data[0]

        for frame_nr in range(data.shape[0]):
            configuration = data[frame_nr, :]
            dq = compute_dq(prev_conf, configuration, dt) if frame_nr > 0 else np.zeros(robot.model.nv)
            pin.forwardKinematics(robot.model, robot.data, configuration, dq)
            pin.updateFramePlacements(robot.model, robot.data)

            root_tf = robot.data.oMi[1]
            pos_root_world = root_tf.translation
            rot_root_world = np.roll(R.from_matrix(root_tf.rotation).as_quat(), 1)
            v_root_world = dq[:3]
            w_root_world = dq[3:6]

            feature_row = list(pos_root_world) + list(rot_root_world) + list(v_root_world) + list(w_root_world)
            
            T_world_from_base = robot.data.oMi[1]
            R_world_from_base = T_world_from_base.rotation
            T_base_from_world = T_world_from_base.inverse()
            R_base_from_world = R_world_from_base.T
 
            velocity_label_xyz_local = R_base_from_world @ v_root_world
            velocity_angular_xyz_local = R_base_from_world @ w_root_world

            label_row = list(velocity_label_xyz_local[:2]) + [velocity_angular_xyz_local[2]]

            for frame_name in frame_names:
                frame_id = robot.model.getFrameId(frame_name)
                tf = robot.data.oMf[frame_id]
                pos_body_world = tf.translation
                pos_body_local = T_base_from_world.act(pos_body_world)

                R_world_from_body = tf.rotation
                R_base_from_body = R_base_from_world @ R_world_from_body
                rot_body_local = np.roll(R.from_matrix(R_base_from_body).as_quat(), 1)

                v = pin.getFrameVelocity(robot.model, robot.data, frame_id, pin.ReferenceFrame.WORLD)
                v_linear_local = R_base_from_world @ v.linear
                v_angular_local = R_base_from_world @ v.angular

                feature_row += list(pos_body_local) + list(rot_body_local) + list(v_linear_local) + list(v_angular_local)

            feature_data_out.append(feature_row)
            if frame_nr > 0:
                label_data_out.append(label_row)

            prev_conf = configuration


        label_data_out.append(np.zeros(3).tolist())  # append zero velocity for last frame

        feature_out_path = os.path.join(feature_out_folder, f"{file_name}_feature.csv")
        label_out_path = os.path.join(label_out_folder, f"{file_name}_label.csv")
        with open(feature_out_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(feature_header)
            writer.writerows(feature_data_out)

        with open(label_out_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(label_header)
            writer.writerows(label_data_out)

        labels_np = np.array(label_data_out, dtype=float)   # shape (T, 3): [linear_x, linear_y, angular_z]
        labels_filter = np.ones_like(labels_np)
        labels_filter[:,0] = butter_lowpass_filtfilt(labels_np[:,0], dt=dt, cutoff_hz=.5, order=2)
        labels_filter[:,1] = labels_np[:,1]
        labels_filter[:,2] = butter_lowpass_filtfilt(labels_np[:,2], dt=dt, cutoff_hz=.5, order=2)
        label_data_out = labels_filter.tolist()    

        if args.plot:
    
            
            T = labels_np.shape[0]
            t = np.arange(T) * dt


            # PLOTTING FRAMES HARDCODED, MAKE SURE WITHIN RANGE OF DATA IF RUNNING WITH PLOTS
            # 1) linear_x vs time
            plt.figure()
            plt.plot(t[:1200], labels_np[:1200, 0], color='red')
            plt.plot(t[:1200], labels_filter[:1200, 0])
            plt.xlabel("Time (s)")
            plt.ylabel("linear_x (m/s)")
            plt.title(f"{file_name}: linear_x vs time")
            plt.tight_layout()
            plt.savefig(os.path.join(label_out_folder, f"{file_name}_linear_x_vs_time_filter5.png"))
            plt.show()

            # 2) linear_y vs time
            plt.figure()
            plt.plot(t[:1200], labels_np[:1200, 1], color='red')
            plt.plot(t[:1200], labels_filter[:1200, 1])
            plt.xlabel("Time (s)")
            plt.ylabel("linear_y (m/s)")
            plt.title(f"{file_name}: linear_y vs time")
            plt.tight_layout()
            plt.savefig(os.path.join(label_out_folder, f"{file_name}_linear_y_vs_time_filter5.png"))
            plt.show()

            # 3) angular_z vs time
            plt.figure()
            plt.plot(t[:1200], labels_np[:1200, 2], color='red')
            plt.plot(t[:1200], labels_filter[:1200, 2])
            plt.xlabel("Time (s)")
            plt.ylabel("angular_z (rad/s)")
            plt.title(f"{file_name}: angular_z vs time")
            plt.tight_layout()
            plt.savefig(os.path.join(label_out_folder, f"{file_name}_angular_z_vs_time_filter5.png"))
            plt.show()
    
        print(f"✅ Saved {feature_out_path} ({len(feature_data_out)} frames, {len(feature_header)} columns)")
        print(f"✅ Saved {label_out_path} ({len(label_data_out)} frames, {len(label_header)} columns)")