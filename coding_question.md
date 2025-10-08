# Example of motion clipping
- https://github.com/Ribosome-rbx/Motion-Matching-for-Human-Skeleton/blob/main/data/mocap/lafan1_stop/walk1_subject1_stop.bvh

The data is trunctuated from `walk1_subject1.bvh` starting from line 424 to 494. It's viewd as the **stop motion**. The original file consists 7k+ frames. Other examples are in `/mocap` folder. 

# Data formart
- Lafan1: 
    - `.bvh` file
    - Tree structure
    - Robot Pose (69D)
        - base `[x, y, z, Rz, Ry, Rx]` (6D)
        - joint `[Rz, Ry, Rx]` (3D * **21**) <font color="red">**(Unitree G1: 29 DOF)**</font>.

- [Omniretarget:](https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset#data-format)
    - `.npy` file
    - Robot Pose (36D)
        - Floating Base [qw, qx, qy, qz, x, y, z] (7D)
        - Joint Positions (29D) (**order?**)

- GMR
    - dictionary: 
    `{
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }`
    - Using official scripts [`extract.py`](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/lafan1/extract.py) to preprocess to get **quaternion, postion and parents**. 
    - [Forward kinematics](https://github.com/YanjieZe/GMR/blob/master/general_motion_retargeting/utils/lafan1.py#L18) deals with tree stucture and offset to get a global pose and orientation.

- [An unofficial retargeted dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
    - The order of the joints is defined.
# Reference

| Project / Repository                                                                                                      | Purpose / Use                                                          | Key Scripts / Usage Notes                                                                                       |
| ------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **[ubisoft/ubisoft-laforge-animation-dataset](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)**             | Official LAFAN1 dataset + example preprocessing and evaluation scripts | Includes `evaluate.py`, `evaluate_test.py` for decompression, BVH format validation, and statistical evaluation |
| **[jihoonerd/Robust-Motion-In-betweening](https://github.com/jihoonerd/Robust-Motion-In-betweening)**                     | Implementation for motion in-betweening using LAFAN1                   | Contains `evaluate.py` for LAFAN1 dataset extraction and verification                                           |
| **[victorqin/motion_inbetweening](https://github.com/victorqin/motion_inbetweening)**                                     | Transformer-based motion interpolation / in-betweening                 | README includes instructions like: “Download LAFAN1 dataset, extract `lafan1.zip` to `datasets/lafan1`”         |
| **[hlcdyy/pan-motion-retargeting](https://github.com/hlcdyy/pan-motion-retargeting)**                                     | Motion retargeting using LAFAN1 and other datasets                     | Script `data_preprocess/Lafan1_and_dog/extract.py` handles LAFAN1 data extraction and preprocessing             |
| **[Ribosome-rbx/Motion-Matching-for-Human-Skeleton](https://github.com/Ribosome-rbx/Motion-Matching-for-Human-Skeleton)** | Motion matching pipeline trained on LAFAN1                             | Includes support for running experiments on the LAFAN1 dataset                                                  |
| **[wangchek/MuscleVAE_](https://github.com/wangchek/MuscleVAE_)**                                                         | Uses LAFAN1 as motion data for training / input                        | Provides scripts like `build_motion_dataset.py` for motion preprocessing and dataset construction               |
