# Using retargeted data
## Lafan1_Retargeting_Dataset
- First clone the Lafan1_Retargeting_Dataset repo to your workspace. I only uploaded csv file to our repo in `g1_retargeted _dataset`.
- The order of configuration is given, root xyz/quaternion + 29 * joint angles, **36** numbers in a row.
- **@Jak**: the velocity label file `bvh_parser` needs to deal with quaternion first. And if any, erase all the other parts that deal with body link pos/ori first because the rest of the data are joint angles.
- **@Sangwoo**: the clip script `clip_bvh_batch` needs to be changed a little to deal with the corresponding csv format. These changes are simple.
## Hands movement
- **@Hanqing**: I think we could set `shoudler_pitch_joint`,`elblow_joint` and all the `wrist_joint` to their default values. This is not only uesful in removing data with big hands movement, but also could be a way to focus on locomotion even for the clean data.

## Kinematics
- To get the all the body positions we may use kinematics defined in `pinocchio`. However GMR used modified joint to deal with bvh file, so it sounds more convenient for us to get away with this transformation. Good things for us.

## Data preprocessing
1. **@all**: define start and end frame for segments as **@sangwoo** suggested.
2. **@all**: Apply FK to get body positions/orientations, and then calculate velocities. Creating seperate branches may be a good idea in that case. The complete state set consists of:

    | Numbers| State Variables    | Dimension |
    |:--:| :---------:| :----------:|
    |1| Root positions      | 3       |
    |2| Root orientations   | 3/4     |
    |3| Root linear velocities| 3     |
    |4| Root angular velocities| 3     |
    |5| Body positions      | 3 * N   |
    |6| Body orientations   | 3/4 * N |
    |7| Body linear velocities     | 3 * N |
    |8| Body angular velocities| 3 * N |
    |9| Joint angles        | J       |
    |10| Joint velocities    | J       |

    Although we may not be able to use them all, calculating it first will be good for experiments. We will calculate it from csv file step by step. That means we can build on `bvh_parser.py`

## Dataloader
- I will look at this as well to provide more details. We could start by state represtantation: 1+2+3+5+6 which is the same as beyondmimic.

## Experiments
### Hanqing: 
- Hands movement: Set some joint angles to fixed values could be a bad choice. **It makes the whole move more weird and cause collision.** Potentially we can restrict the `elbow` and `shoulder_roll` joints. But even in the original regtarted data there is collision: 
![collision](/img/collision.png "collision in original retarget")

- @Jak: velocity label: `frame_tf = self.robot.data.oMf[frame_id]
            joint_tf = self.robot.data.oMi[parent_joint_id]`
            
    These lines in #70 in `rerun_visualize.py` define R = 3 by 3 rotation matrix and p = 3 by 1 translation vector. They define joint frame and body frame and calculate joint to body transform. **We may want to find transform for the `root_joint` first.** and calculate everything relative to that. It should be more stable compared to the hip joint in bvh.
- Built the initial dataset and mirror transform in `/model`.
    - `dataset.py`: convert csv data and label to tensor, sample data of length `seq_len`.
    - `transform.py`: define mirror transfrom to swap left and right, take the negative of waist roll angle (to be checked)
- Built two CVAE models in `/model/models.py`
    - `TrajCVAE` uses gated recurrent unit (GRU) to input and output trajectories.
    - `PoseCVAE` encodes global condition and decodes based on condition in each frame. 

- Todo: 
    1. add `RandomsWeightedSampler` to dataloader to ensure every segment has the same probability to be loaded.
    2. fill the exact names of column.
    3. write train.py

### 10.22
Got the initial result. The body seems to be torn apart, which means the generated body positions couldn't make a whole body.
![initial_result](/img/initial_result.png)

### 10.24
After carefully reviewed about the model part, I decided to use the teacher forcing technique in VAE training. And the result is much better, although it still has some artifact like seperate body parts.
![teacher_forcing](/img/teacher_forcing.png)
