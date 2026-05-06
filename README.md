# Generating Human Priors for Mimic Control

Keywords: Humanoid Locomotion, Motion Generation, Motion Tracking, Reinforcement Learning, Sim-to-Real

## Project Overview

This project presents a two-stage pipeline for generating **natural, human-like** locomotion with **real-time joystick control** for the **Unitree G1 humanoid robot**. The pipeline bridges the gap between human motion capture (Mocap) data and dynamically feasible robot control.

1. Command-conditioned offline **motion generation**
2. RL-based **motion tracking** with joystick commands

(Video) regular walking poliy vs ours

## Stage 1: Motion Generation

<video src="img/motion_gen_video.mp4" controls></video>

This **autoregressive** generative model can predict robot trajectories with natural transitions conditioned on future joystick commands over a **long horizon**. The first stage serves as a **data engine** to obtain the robot's response to joystick command labels.
- **Dataset**: Uses the **[LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)** for training. The data is specifically retargeted to the G1 humanoid's joint limits and link proportions.
- **Motion Representation**:
  - 36-dimensional state representation $S_t$:
    - Robot base velocity and orientation `(7)`
    - Joint angles `(29)`
  - 3-dimensional joystick command $V_t$, extracted and filtered from original motion data.
    - $V_t = [v_x, v_y, v_\omega]$
- **Model Architecture**: A lightweight, kinematics-only, GRU (Gated Recurrent Unit)-based **sequence prediction** model that encodes past states and predicts future states conditioned on future joystick commands.

  $$S_{t:t+20} = f(S_{t-10:t}, V_{t:t+20})$$

- **Real-Time Inference and Visualization**: Capable of processing joystick inputs instantly to generate responsive motion in real time with low latency. Integrated with the **[Rerun SDK](https://rerun.io/)** for real-time visualization of the robot's trajectory.

## Stage 2: Motion Tracking

We use 10-minute Stage 1-generated robot trajectories at 30 Hz, together with joystick command labels, as training data for the second stage.

The second stage follows the **[BeyondMimic Motion Tracking](https://github.com/HybridRobotics/whole_body_tracking/tree/main)** framework with modified observations. The original observation is:

$$O_t=[\psi, e_{anchor}, \nu_{imu}, \theta-\theta_0, \dot{\theta}, a_{last}]$$

- **Reference-free**: Since we're not strictly following the generated motion, the reference motion $\psi$ and anchor error $e_{anchor}$ are no longer needed.
- **Joystick input**: Instead, the policy should respond to joystick command $V_t$.
- **Temporal information**: Removing the reference motion can lead to insufficient information in the motion tracking task, so we augment the policy observation with **history**. Each term in the single-frame observation is stacked over a history window of H frames.

The final observation is:

$$o_t=[V_{t}, \nu_{imu}, \theta-\theta_0, \dot{\theta}, a_{last}]$$
$$O_t = [o_{t-H+1}, \dots, o_t]$$

## Code Structure
- `model/dataset.py`: Dataset loader and normalization utilities.
- `model/inference_rt.py`: Real-time autoregressive inference with joystick input and visualization.
- `model/joystick.py`: Joystick input interface.
- `model/models.py`: Motion generation model definitions.
- `model/rerun_visualize.py`: Rerun-based robot visualization utilities.
- `model/train.py`: Training entry point for the motion generator.

## Dataset Preparation

```text
dataset/
├── data_joint/
│   ├── walk/
│   └── run/
├── data_feature/
│   ├── walk/
│   └── run/
├── data_label/
│   ├── walk/
│   └── run/
├── g1_retargeted_dataset/
└── extract_features_labels.py
```

Each folder under `data_*` is organized by motion type, such as `walk/` and `run/`.

1. Download the **[LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)** to `dataset/g1_retargeted_dataset/`.
2. Clip the locomotion segments from `dataset/g1_retargeted_dataset/` into `dataset/data_joint/` under specific motion type.
3. Run `dataset/extract_features_labels.py` to extract the motion representation into `dataset/data_feature/` and the joystick command labels into `dataset/data_label/`.


## References

[2] A. Martinez, M. Black, and J. Romero, "On Human Motion Prediction Using Recurrent Neural Networks," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017.

[4] H. Y. Ling, F. Zinno, G. Cheng, and M. Van de Panne, "Character Controllers Using Motion VAEs," *ACM Transactions on Graphics*, vol. 39, no. 4, Art. 40, Aug. 2020, 12 pages.

[8] Q. Liao et al., "BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion," 2024.

[10] J. Harvey et al., "LAFAN1: A High-Quality Motion Capture Dataset for Animation Research," 2020.
