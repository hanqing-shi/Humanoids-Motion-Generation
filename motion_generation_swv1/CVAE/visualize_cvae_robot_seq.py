#!/usr/bin/env python3
import os
import time
import argparse
import numpy as np
import torch
import rerun as rr
import pinocchio as pin
import trimesh

# ======== must match training ========
HORIZON   = 30
D_CONF    = 36
COND_DIM  = D_CONF + 3
LATENT_DIM= 32

# ======== CVAE (decoder + encode for future GT if needed later) ========
class MotionCVAE(torch.nn.Module):
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
        self.d_conf  = D_CONF

        # encoder
        enc_in = cond_dim + future_dim
        enc = []
        dim = enc_in
        for _ in range(depth):
            enc += [torch.nn.Linear(dim, hidden_dim), torch.nn.ReLU()]
            dim = hidden_dim
        self.encoder_body = torch.nn.Sequential(*enc)
        self.to_mu     = torch.nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = torch.nn.Linear(hidden_dim, latent_dim)

        # decoder
        dec_in = cond_dim + latent_dim
        dec = []
        dim = dec_in
        for _ in range(depth):
            dec += [torch.nn.Linear(dim, hidden_dim), torch.nn.ReLU()]
            dim = hidden_dim
        dec += [torch.nn.Linear(hidden_dim, future_dim)]
        self.decoder_body = torch.nn.Sequential(*dec)

    def decode(self, cond, z):
        d_in = torch.cat([cond, z], dim=-1)            # (B, 39+latent)
        flat = self.decoder_body(d_in)                 # (B, 1080)
        seq  = flat.view(-1, self.horizon, self.d_conf)# (B, 30, 36)
        return seq

# ======== URDF renderer with prefix (pred/gt 분리용) ========
class RerunURDFPrefixed:
    def __init__(self, robot_type: str, path_prefix: str = "pred"):
        self.prefix = path_prefix
        self.name   = robot_type

        if robot_type == 'g1':
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
        elif robot_type == 'h1_2':
            self.robot = pin.RobotWrapper.BuildFromURDF(
                'robot_description/h1_2/h1_2_wo_hand.urdf',
                'robot_description/h1_2',
                pin.JointModelFreeFlyer()
            )
            self.Tpose = np.array([
                0,0,1.02,0,0,0,1,
                0,-0.15,0,0.3,-0.15,0,
                0,-0.15,0,0.3,-0.15,0,
                0,
                0, 1.57,0,1.57,0,0,0,
                0,-1.57,0,1.57,0,0,0
            ], dtype=np.float32)
        elif robot_type == 'h1':
            self.robot = pin.RobotWrapper.BuildFromURDF(
                'robot_description/h1/h1.urdf',
                'robot_description/h1',
                pin.JointModelFreeFlyer()
            )
            self.Tpose = np.array([
                0,0,1.03,0,0,0,1,
                0,0,-0.15,0.3,-0.15,
                0,0,-0.15,0.3,-0.15,
                0,
                0, 1.57,0,1.57,
                0,-1.57,0,1.57
            ], dtype=np.float32)
        else:
            raise ValueError("Invalid robot type")

        self.link2mesh = self._get_link2mesh()
        self._load_visual_mesh()
        self.update()

    def _get_link2mesh(self):
        link2mesh = {}
        for visual in self.robot.visual_model.geometryObjects:
            mesh = trimesh.load_mesh(visual.meshPath)
            name = visual.name[:-2]
            mesh.visual = trimesh.visual.ColorVisuals()
            mesh.visual.vertex_colors = visual.meshColor
            link2mesh[name] = mesh
        return link2mesh

    def _load_visual_mesh(self):
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
                f'{self.prefix}/urdf_{self.name}/{parent_joint_name}',
                rr.Transform3D(
                    translation=joint_tf.translation,
                    mat3x3=joint_tf.rotation,
                    axis_length=0.01
                )
            )
            relative_tf = joint_tf.inverse() * frame_tf
            mesh.apply_transform(relative_tf.homogeneous)
            rr.log(
                f'{self.prefix}/urdf_{self.name}/{parent_joint_name}/{frame_name}',
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
                f'{self.prefix}/urdf_{self.name}/{parent_joint_name}',
                rr.Transform3D(
                    translation=joint_tf.translation,
                    mat3x3=joint_tf.rotation,
                    axis_length=0.01
                )
            )

# ======== main ========
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--robot_type',   type=str, default='g1')
    ap.add_argument('--csv_path',     type=str, default='retarget_out_vel/walk1_subject1_mid_vel.csv')
    ap.add_argument('--model_path',   type=str, default='motion_cvae.pt')
    ap.add_argument('--device',       type=str, default='cpu')
    ap.add_argument('--mode',         type=str, default='both', choices=['pred','gt','both'],
                    help='what to visualize per window')
    ap.add_argument('--stride',       type=int, default=5, help='slide step between windows')
    ap.add_argument('--z_mode',       type=str, default='zero', choices=['zero','random'],
                    help='latent sampling mode for predictions')
    ap.add_argument('--seed',         type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load model
    ckpt = torch.load(args.model_path, map_location=device)
    model = MotionCVAE(
        cond_dim=COND_DIM,
        future_dim=HORIZON*D_CONF,
        latent_dim=ckpt.get("latent_dim", LATENT_DIM)
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # load csv
    data = np.genfromtxt(args.csv_path, delimiter=',', skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    T = data.shape[0]

    # init rerun & two renderers (pred/gt)
    rr.init('CVAE_RobotSeq', spawn=True)
    rr.log('', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    pred_renderer = RerunURDFPrefixed(args.robot_type, path_prefix="pred") if args.mode in ('pred','both') else None
    gt_renderer   = RerunURDFPrefixed(args.robot_type, path_prefix="gt")   if args.mode in ('gt','both')   else None

    time_idx = 0

    # slide over the whole CSV
    for t in range(0, max(0, T - HORIZON - 1), args.stride):
        q_t   = data[t, :D_CONF]
        vel_t = data[t, D_CONF:D_CONF+3]

        # get prediction sequence
        if args.mode in ('pred','both'):
            cond_vec = np.concatenate([q_t, vel_t]).astype(np.float32)
            cond = torch.from_numpy(cond_vec).unsqueeze(0).to(device)  # (1,39)

            if args.z_mode == 'zero':
                z = torch.zeros(1, model.latent_dim, device=device)
            else:
                z = torch.randn(1, model.latent_dim, device=device)

            pred_seq = model.decode(cond, z).squeeze(0).cpu().numpy()  # (30,36)

        # get ground-truth sequence
        if args.mode in ('gt','both'):
            gt_seq = data[t+1:t+1+HORIZON, :D_CONF]  # (30,36)

        # play this window
        for k in range(HORIZON):
            rr.set_time_sequence('frame_nr', time_idx)
            if pred_renderer is not None:
                pred_renderer.update(pred_seq[k])
            if gt_renderer is not None:
                gt_renderer.update(gt_seq[k])
            time_idx += 1
            time.sleep(0.01)  # faster playback. use 0.03 for ~real-time 30Hz

if __name__ == "__main__":
    main()
