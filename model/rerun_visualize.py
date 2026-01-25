import argparse
import time
import numpy as np
import pinocchio as pin
import rerun as rr
import trimesh
from scipy.spatial.transform import Rotation as R

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, help="File name", default='0101_TrajCVAE_000_b31')
    parser.add_argument('--robot_type', type=str, help="Robot type", default='g1')
    return parser.parse_args()

class RerunBody():
    def __init__(self, robot_type):
        self.robot_type = robot_type
        frame_names = [
            'pelvis', 'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link', 'left_knee_link',
            'left_ankle_pitch_link', 'left_ankle_roll_link', 'pelvis_contour_link', 'right_hip_pitch_link',
            'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_pitch_link',
            'right_ankle_roll_link', 'waist_yaw_link', 'waist_roll_link', 'torso_link', 'head_link',
            'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link',
            'left_elbow_link', 'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link',
            'left_rubber_hand', 'logo_link', 'right_shoulder_pitch_link', 'right_shoulder_roll_link',
            'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_roll_link', 'right_wrist_pitch_link',
            'right_wrist_yaw_link', 'right_rubber_hand'
        ]
        feature_header = [
        'root_x', 'root_y', 'root_z',
        'root_qw', 'root_qx', 'root_qy', 'root_qz',
        'root_vx', 'root_vy', 'root_vz',
        'root_wx', 'root_wy', 'root_wz'
        ]
        for f in frame_names:
            feature_header += [
                f'{f}_pos_x', f'{f}_pos_y', f'{f}_pos_z',
                f'{f}_ori_w', f'{f}_ori_x', f'{f}_ori_y', f'{f}_ori_z',
                #f'{f}_vel_x', f'{f}_vel_y', f'{f}_vel_z',
                #f'{f}_angvel_x', f'{f}_angvel_y', f'{f}_angvel_z'
            ]
        self.feature_header = feature_header
        self.frame_names = frame_names
        self.name2idx = {name: i for i, name in enumerate(feature_header)}

        self.robot = pin.RobotWrapper.BuildFromURDF(
            './dataset/g1_retargeted_dataset/g1/g1_29dof_rev_1_0.urdf', './dataset/g1_retargeted_dataset/g1', pin.JointModelFreeFlyer()
        )

        self.link2mesh = self._load_mesh()
        self._log_mesh_once()

    # -------------------------
    def _load_mesh(self):
        link2mesh = {}
        for visual in self.robot.visual_model.geometryObjects:
            frame_name = visual.name[:-2]
            mesh = trimesh.load_mesh(visual.meshPath)
            mesh.visual = trimesh.visual.ColorVisuals()
            mesh.visual.vertex_colors = visual.meshColor
            link2mesh[frame_name] = mesh
        print(f"✅ Loaded {len(link2mesh)} meshes")
        return link2mesh

    # -------------------------
    def _log_mesh_once(self):
        for f, mesh in self.link2mesh.items():
            rr.log(
                f"{self.robot_type}/{f}/mesh",
                rr.Mesh3D(
                    vertex_positions=mesh.vertices,
                    triangle_indices=mesh.faces,
                    vertex_normals=mesh.vertex_normals,
                    vertex_colors=mesh.visual.vertex_colors,
                ),
                static=True,
            )

    # -------------------------
    def update(self, configuration):
        # ==== root ====
        pos_root = np.array([
            configuration[self.name2idx["root_x"]],
            configuration[self.name2idx["root_y"]],
            configuration[self.name2idx["root_z"]],
        ])

        quat_root = np.array([
            configuration[self.name2idx["root_qx"]],
            configuration[self.name2idx["root_qy"]],
            configuration[self.name2idx["root_qz"]],
            configuration[self.name2idx["root_qw"]],
        ])
        R_root = R.from_quat(quat_root).as_matrix()

        rr.log(
            f"{self.robot_type}/root",
            rr.Transform3D(
                translation=pos_root,
                mat3x3=R_root,
                axis_length=0.05
            )
        )

        # ==== body ====
        for f in self.frame_names:
            if f not in self.link2mesh:
                continue

            pos = np.array([
                configuration[self.name2idx[f"{f}_pos_x"]],
                configuration[self.name2idx[f"{f}_pos_y"]],
                configuration[self.name2idx[f"{f}_pos_z"]],
            ])

            quat = np.array([
                configuration[self.name2idx[f"{f}_ori_x"]],
                configuration[self.name2idx[f"{f}_ori_y"]],
                configuration[self.name2idx[f"{f}_ori_z"]],
                configuration[self.name2idx[f"{f}_ori_w"]],
            ])
            R_link = R.from_quat(quat).as_matrix()

            pos_world = pos_root + R_root @ pos
            R_world = R_root @ R_link

            rr.log(
                f"{self.robot_type}/{f}",
                rr.Transform3D(
                    translation=pos_world,
                    mat3x3=R_world
                )
            )

class RerunJoint():
    def __init__(self, robot_type):
        self.name = robot_type
        match robot_type:
            case 'g1':
                self.robot = pin.RobotWrapper.BuildFromURDF('./dataset/g1_retargeted_dataset/g1/g1_29dof_rev_1_0.urdf', './dataset/g1_retargeted_dataset/g1', pin.JointModelFreeFlyer())

                # self.Tpose = np.array([0,0,0.785,0,0,0,1,
                #                        -0.15,0,0,0.3,-0.15,0,
                #                        -0.15,0,0,0.3,-0.15,0,
                #                        0,0,0,
                #                        0, 1.57,0,1.57,0,0,0,
                #                        0,-1.57,0,1.57,0,0,0]).astype(np.float32) # pose with arms up
                
                self.Tpose = np.array([
                                0.0, 0.0, 0.76,
                                0.0, 0.0, 0.0, 1.0,
                                -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
                                -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
                                0.0, 0.0, 0.0,
                                0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
                                0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0]).astype(np.float32)
            case _:
                print(robot_type)
                raise ValueError('Invalid robot type')
        
        self.link2mesh = self.get_link2mesh()
        self.load_visual_mesh()
        self.update()
    
    def get_link2mesh(self):
        link2mesh = {}
        for visual in self.robot.visual_model.geometryObjects:
            mesh = trimesh.load_mesh(visual.meshPath)
            name = visual.name[:-2]
            mesh.visual = trimesh.visual.ColorVisuals()
            mesh.visual.vertex_colors = visual.meshColor
            link2mesh[name] = mesh
        return link2mesh
   
    def load_visual_mesh(self):       
        self.robot.framesForwardKinematics(pin.neutral(self.robot.model))
        for visual in self.robot.visual_model.geometryObjects:
            frame_name = visual.name[:-2]
            mesh = self.link2mesh[frame_name]
            frame_id = self.robot.model.getFrameId(frame_name)
            parent_joint_id = self.robot.model.frames[frame_id].parent
            parent_joint_name = self.robot.model.names[parent_joint_id]
            
            frame_tf = self.robot.data.oMf[frame_id]
            joint_tf = self.robot.data.oMi[parent_joint_id]
            rr.log(f'urdf_{self.name}/{parent_joint_name}',
                   rr.Transform3D(translation=joint_tf.translation,
                                  mat3x3=joint_tf.rotation,
                                  axis_length=0.01))
            
            relative_tf = joint_tf.inverse() * frame_tf
            mesh.apply_transform(relative_tf.homogeneous)
            rr.log(f'urdf_{self.name}/{parent_joint_name}/{frame_name}',
                   rr.Mesh3D(
                       vertex_positions=mesh.vertices,
                       triangle_indices=mesh.faces,
                       vertex_normals=mesh.vertex_normals,
                       vertex_colors=mesh.visual.vertex_colors,
                       albedo_texture=None,
                       vertex_texcoords=None,
                   ),
                   static=True)
    
    def update(self, configuration = None):
        self.robot.framesForwardKinematics(self.Tpose if configuration is None else configuration)
        for visual in self.robot.visual_model.geometryObjects:
            frame_name = visual.name[:-2]
            frame_id = self.robot.model.getFrameId(frame_name)
            parent_joint_id = self.robot.model.frames[frame_id].parent
            parent_joint_name = self.robot.model.names[parent_joint_id]
            joint_tf = self.robot.data.oMi[parent_joint_id]
            rr.log(f'urdf_{self.name}/{parent_joint_name}',
                   rr.Transform3D(translation=joint_tf.translation,
                                  mat3x3=joint_tf.rotation,
                                  axis_length=0.01))

if __name__ == "__main__":
    args = parse_cli()

    rr.init(
        'Reviz', 
        spawn=True
    )
    rr.log('', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    csv_files = f"./results/{args.file_name}.csv"
    data = np.genfromtxt(csv_files, delimiter=',')
    
    rerun_urdf = RerunJoint(args.robot_type) if data.shape[1] == 36 else RerunBody(args.robot_type)
    print('model type:', 'joint' if data.shape[1] == 36 else 'body')
    for frame_nr in range(data.shape[0]):
        rr.set_time_sequence('frame_nr', frame_nr)
        configuration = data[frame_nr, :]
        rerun_urdf.update(configuration)
        time.sleep(0.03)
