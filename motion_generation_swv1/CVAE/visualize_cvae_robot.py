#!/usr/bin/env python3
import os
import time
import numpy as np
import torch
import argparse
import rerun as rr
import pinocchio as pin
import trimesh

# ======== import model pieces from train_motion_cvae.py ========

# You MUST make sure these match what you used in training
HORIZON = 30
D_CONF = 36
COND_DIM = D_CONF + 3
LATENT_DIM = 32  # must match LATENT_DIM in train_motion_cvae.py

class MotionCVAE(torch.nn.Module):
    """
    Same CVAE definition used during training.
    - cond: [q_t (36), vx,vy,wz] -> (39,)
    - future: flattened future rollout (30*36,)
    """
    def __init__(self,
                 cond_dim=COND_DIM,
                 future_dim=HORIZON*D_CONF,
                 latent_dim=LATENT_DIM,
                 hidden_dim=512,
                 depth=4):
        super().__init__()
        self.cond_dim = cond_dim
        self.future_dim = future_dim
        self.latent_dim = latent_dim
        self.horizon = HORIZON
        self.d_conf = D_CONF

        # encoder
        enc_in_dim = cond_dim + future_dim
        enc_layers = []
        dim = enc_in_dim
        for _ in range(depth):
            enc_layers.append(torch.nn.Linear(dim, hidden_dim))
            enc_layers.append(torch.nn.ReLU())
            dim = hidden_dim
        self.encoder_body = torch.nn.Sequential(*enc_layers)
        self.to_mu     = torch.nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = torch.nn.Linear(hidden_dim, latent_dim)

        # decoder
        dec_in_dim = cond_dim + latent_dim
        dec_layers = []
        dim = dec_in_dim
        for _ in range(depth):
            dec_layers.append(torch.nn.Linear(dim, hidden_dim))
            dec_layers.append(torch.nn.ReLU())
            dim = hidden_dim
        dec_layers.append(torch.nn.Linear(hidden_dim, self.future_dim))
        self.decoder_body = torch.nn.Sequential(*dec_layers)

    def decode(self, cond, z):
        # cond: (B,39)
        # z:    (B,latent_dim)
        d_in = torch.cat([cond, z], dim=-1)  # (B, 39+latent_dim)
        future_hat_flat = self.decoder_body(d_in)  # (B,1080)
        future_hat_seq = future_hat_flat.view(-1, self.horizon, self.d_conf)  # (B,30,36)
        return future_hat_seq  # (B,30,36)

# ======== RerunURDF class (your renderer) ========

class RerunURDF():
    def __init__(self, robot_type):
        self.name = robot_type
        match robot_type:
            case 'g1':
                self.robot = pin.RobotWrapper.BuildFromURDF(
                    'robot_description/g1/g1_29dof_rev_1_0.urdf',
                    'robot_description/g1',
                    pin.JointModelFreeFlyer()
                )
                self.Tpose = np.array([
                    0,0,0.785,0,0,0,1,
                    -0.15,0,0,0.3,-0.15,0,
                    -0.15,0,0,0.3,-0.15,0,
                    0,0,0,
                    0, 1.57,0,1.57,0,0,0,
                    0,-1.57,0,1.57,0,0,0
                ], dtype=np.float32)

            case 'h1_2':
                self.robot = pin.RobotWrapper.BuildFromURDF(
                    'robot_description/h1_2/h1_2_wo_hand.urdf',
                    'robot_description/h1_2',
                    pin.JointModelFreeFlyer()
                )
                assert self.robot.model.nq == 7 + 12 + 1 + 14
                self.Tpose = np.array([
                    0,0,1.02,0,0,0,1,
                    0,-0.15,0,0.3,-0.15,0,
                    0,-0.15,0,0.3,-0.15,0,
                    0,
                    0, 1.57,0,1.57,0,0,0,
                    0,-1.57,0,1.57,0,0,0
                ], dtype=np.float32)

            case 'h1':
                self.robot = pin.RobotWrapper.BuildFromURDF(
                    'robot_description/h1/h1.urdf',
                    'robot_description/h1',
                    pin.JointModelFreeFlyer()
                )
                assert self.robot.model.nq == 7 + 10 + 1 + 8
                self.Tpose = np.array([
                    0,0,1.03,0,0,0,1,
                    0,0,-0.15,0.3,-0.15,
                    0,0,-0.15,0.3,-0.15,
                    0,
                    0, 1.57,0,1.57,
                    0,-1.57,0,1.57
                ], dtype=np.float32)

            case _:
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
            rr.log(
                f'urdf_{self.name}/{parent_joint_name}',
                rr.Transform3D(
                    translation=joint_tf.translation,
                    mat3x3=joint_tf.rotation,
                    axis_length=0.01
                )
            )
            
            relative_tf = joint_tf.inverse() * frame_tf
            mesh.apply_transform(relative_tf.homogeneous)
            rr.log(
                f'urdf_{self.name}/{parent_joint_name}/{frame_name}',
                rr.Mesh3D(
                    vertex_positions=mesh.vertices,
                    triangle_indices=mesh.faces,
                    vertex_normals=mesh.vertex_normals,
                    vertex_colors=mesh.visual.vertex_colors,
                    albedo_texture=None,
                    vertex_texcoords=None,
                ),
                static=True
            )
    
    def update(self, configuration=None):
        self.robot.framesForwardKinematics(
            self.Tpose if configuration is None else configuration
        )
        for visual in self.robot.visual_model.geometryObjects:
            frame_name = visual.name[:-2]
            frame_id = self.robot.model.getFrameId(frame_name)
            parent_joint_id = self.robot.model.frames[frame_id].parent
            parent_joint_name = self.robot.model.names[parent_joint_id]
            joint_tf = self.robot.data.oMi[parent_joint_id]
            rr.log(
                f'urdf_{self.name}/{parent_joint_name}',
                rr.Transform3D(
                    translation=joint_tf.translation,
                    mat3x3=joint_tf.rotation,
                    axis_length=0.01
                )
            )

# ======== helper: sample one rollout from model ========

@torch.no_grad()
def generate_rollout_from_cvae(model, q_t, vel_t, n_samples=1, device="cpu"):
    """
    q_t:   (36,) numpy
    vel_t: (3,)  numpy  -> [vx, vy, wz]
    returns: list of rollout(s), each (HORIZON, 36)
    """
    model.eval()
    cond_vec = np.concatenate([q_t, vel_t]).astype(np.float32)  # (39,)
    cond = torch.from_numpy(cond_vec).unsqueeze(0).to(device)   # (1,39)

    outs = []
    for _ in range(n_samples):
        z = torch.randn(1, model.latent_dim, device=device)
        future_seq = model.decode(cond, z)  # (1,30,36)
        outs.append(future_seq.squeeze(0).cpu().numpy())
    return outs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_type', type=str, default='g1')
    parser.add_argument('--csv_path', type=str,
                        default='retarget_out_vel/walk1_subject1_mid_vel.csv')
    parser.add_argument('--model_path', type=str,
                        default='motion_cvae.pt')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--which_sample', type=int, default=0,
                        help='which sampled rollout to visualize (0..n-1)')
    args = parser.parse_args()

    device = torch.device(args.device)

    # 1. load trained CVAE weights
    ckpt = torch.load(args.model_path, map_location=device)
    model = MotionCVAE(
        cond_dim=COND_DIM,
        future_dim=HORIZON*D_CONF,
        latent_dim=ckpt["latent_dim"]
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    # 2. grab an initial state (q_t, vel_t) from CSV first frame
    data = np.genfromtxt(args.csv_path, delimiter=',', skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    # split row -> q / vel
    q_t   = data[0, :D_CONF]               # (36,)
    vel_t = data[0, D_CONF:D_CONF+3]       # (3,)

    # 3. sample rollout(s) from CVAE
    rollouts = generate_rollout_from_cvae(model, q_t, vel_t,
                                          n_samples=3,
                                          device=device)
    traj = rollouts[args.which_sample]     # (30,36)

    # 4. init rerun viewer
    rr.init('CVAE_RobotViz', spawn=True)
    rr.log('', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # 5. create URDF renderer
    rerun_urdf = RerunURDF(args.robot_type)

    # 6. play trajectory through rerun
    for frame_idx in range(traj.shape[0]):
        rr.set_time_sequence('frame_nr', frame_idx)
        configuration = traj[frame_idx, :]  # (36,)
        rerun_urdf.update(configuration)
        time.sleep(0.03)

if __name__ == "__main__":
    main()
