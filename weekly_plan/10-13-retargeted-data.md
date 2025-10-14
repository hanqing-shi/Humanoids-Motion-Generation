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

    | Numbers| State Variables    | Dimention |
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
