import torch
import models
import os
import time
import numpy as np
import argparse
from rerun_visualize import RerunJoint
import rerun as rr
from joystick import JoystickController
from control.g1.safety import G1SafetyFilter ##### ADD THESE IMPORTS ####################### 
from control.g1.tracking.tracker import HybridPDTracker, HybridPDParams ####################

USE_DIAGNOSTICS = True
if USE_DIAGNOSTICS:
    try:
        from control.g1.diagnostics import split_actuated, set_actuated, assert_full_configuration
    except Exception:
        USE_DIAGNOSTICS = False

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TrajCVAE", help="Model class name in models.py")
    parser.add_argument("--past-lenth", type=int, default=10)
    parser.add_argument("--inference-step", type=int, default=300)
    parser.add_argument("--hz", type=float, default=30.0, help="Visualization/control rate (Hz)") ######
    parser.add_argument("--chunk", type=int, default=20, help="Steps sampled per outer frame") #########
    return parser.parse_args()

DEBUG_POKE = False    # Poking robot to make sure joints indexed correctly and sim measuring joints correctly
POKE_J = 0            # 0..28 within the actuated joint vector
POKE_DELTA = 0.3      # rad

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    ModelClass = getattr(models, args.model)

    if args.model == "TrajCVAE":
        model = ModelClass(traj_dim=36, cond_dim=3, teacher_forcing=0, past_lenth=args.past_lenth).to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    print(f"Loaded model: {args.model}")

    ckpt_path = f"checkpoints/{args.model}_best_walk.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print("✅ Model loaded and ready for inference")
    else:
        print(f"⚠️ Warning: Checkpoint not found at {ckpt_path}")

    model.eval()

    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)

    # Defines inition 36D pose
    # pull this over to help with closed loop feedback
    x_init_np = np.array([
        0.0, 0.0, 0.76,
        0.0, 0.0, 0.0, 1.0,
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        0.0, 0.0, 0.0,
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0
    ], dtype=np.float32)

    if USE_DIAGNOSTICS:
        assert_full_configuration(x_init_np, "x_init_np")

    x_init = torch.from_numpy(x_init_np).to(device).unsqueeze(0).unsqueeze(0)  # (1,1,36)

    # Init Joystick
    controller = JoystickController(motion="walk", deadzone=0.1)
    assert controller.joystick is not None, "Joystick not connected. Please connect a joystick and try again."

    # init visualizer USES NEW PARAMETERS
    rr.init("Reviz", spawn=True)
    rr.log("", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rerun_urdf = RerunJoint("g1")

    hz = float(args.hz) ##########
    dt = 1.0 / hz ################
    chunk = int(args.chunk) ######

    # init safety filter
    safety = G1SafetyFilter(dt=dt)

    # Initialize filter state using actuated slice
    if USE_DIAGNOSTICS:
        q_init = split_actuated(x_init_np)
    else:
        q_init = x_init_np[-29:]  # assumes last 29 are actuated joints in your convention
    safety.reset(q_init)

    # init tracker
    params = HybridPDParams(
        alpha=0.4,
        dq_limit=1e6,       # limits will be enforced by safety filter, so set these very high to avoid interference
        ddq_limit=None,     # set acceleration later if needed
        kp_default=60.0,    # stiffness
        kd_default=6.0,     # damping
        use_critical_damping=True,
        output_dq_cmd=True
        )
    
    tracker = HybridPDTracker(n=29, dt=dt, params=params)
    tracker.reset(q_init)

    # Inference Loop
    with torch.no_grad():
        # initial context
        cond_past = torch.zeros((1, args.past_lenth, 3), device=device)
        x_past = x_init.repeat(1, args.past_lenth, 1)  # (1, past_lenth, 36)
        x_start = x_init[:, -1:, :]
        traj_array = []

        for frame in range(args.inference_step):
            # Read joystick commands (body-frame)
            cmd = controller.get_cond_commands(steps=chunk, freq=hz).reshape(1, -1, 3)  # (1,chunk,3)
            cond = torch.from_numpy(cmd).float().to(device)

            # Sample trajectory chunk
            samples = model.sample(cond_past, cond, x_past, x_start)  # expected (1,chunk,36) or similar
            
            last_applied_config = None
            # Visualize each step
            for step in range(samples.shape[1]):
                rr.set_time_sequence("frame_nr", frame * chunk + step) ### Uses chunk param

                configuration = samples[-1, step, :].detach().cpu().numpy()  # (36,)
                if frame == 0 and step == 0:
                    print("configuration dim:", configuration.shape)

                # --- actuated slice ---
                if USE_DIAGNOSTICS:
                    q_model = split_actuated(configuration)
                else:
                    q_model = configuration[-29:]

                # --- safety ---
                q_safe, info = safety.step(q_model)
                if not info.get("ok", True):
                    print("⚠️ Safety triggered:", info.get("reason", "unknown"))

                # tracker - currently uses last command instead of measurement
                if tracker.q_cmd_prev is None:
                    q_meas = q_init.copy()
                    dq_meas = np.zeros_like(q_meas)
                else:
                    q_meas = tracker.q_cmd_prev.copy()
                    dq_meas = tracker.dq_cmd_prev.copy()

                q_cmd, dq_cmd, kp, kd, tau_ff = tracker.step(q_safe, q_meas, dq_meas)

                # --- rebuild full configuration using TRACKED command joints ---
                if USE_DIAGNOSTICS:
                    configuration_safe = set_actuated(configuration, q_cmd)
                else:
                    configuration_safe = configuration.copy()
                    configuration_safe[-29:] = q_cmd

                # Poke
                if DEBUG_POKE and frame == 0 and step == 0:
                    if USE_DIAGNOSTICS:
                        base = x_init_np.copy()
                        q_test = split_actuated(base).copy()
                    else:
                        base = x_init_np.copy()
                        q_test = base[-29:].copy()

                    j = int(POKE_J)
                    if j < 0 or j >= q_test.shape[0]:
                        raise ValueError(f"POKE_J must be 0..{q_test.shape[0]-1}, got {j}")

                    q_test[j] += float(POKE_DELTA)
                    q_test_safe, info2 = safety.step(q_test)
                    if not info2.get("ok", True):
                        print("⚠️ Safety triggered (poke):", info2.get("reason", "unknown"))

                    if USE_DIAGNOSTICS:
                        configuration_safe = set_actuated(base, q_test_safe)
                    else:
                        configuration_safe = base.copy()
                        configuration_safe[-29:] = q_test_safe

                # --- visualize SAFE configuration ---
                last_applied_config = configuration_safe #### need for closed loop
                rerun_urdf.update(configuration_safe)
                time.sleep(dt)

            # Update past information
            # Update past information (CLOSED LOOP)
            cond_past = torch.cat([cond_past, cond], dim=1)[:, -args.past_lenth:, :]
            assert last_applied_config is not None, "No applied configuration recorded"
            applied_np = last_applied_config.astype(np.float32)          # (36,)
            applied_t = torch.from_numpy(applied_np).to(device).view(1, 1, -1)  # (1,1,36)
            x_past = torch.cat([x_past, applied_t], dim=1)[:, -args.past_lenth:, :]
            x_start = applied_t

            traj_array.append(samples)

        # Save results
        traj = torch.cat(traj_array, dim=1).squeeze(0).detach().cpu().numpy()
        controller.close()

        save_path = os.path.join(save_dir, f"{args.model}_rt_safe.csv")
        np.savetxt(save_path, traj, delimiter=",")
        print(f"✅ Sampling complete. Saved to {save_path}")

if __name__ == "__main__":
    args = parse_cli()
    main(args)
