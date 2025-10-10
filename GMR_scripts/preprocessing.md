# Data preprocessing
## Data segments
- **We use 21 files**:
    - walking * 12 
    - run * 4
    - sprint * 2
    - jumps * 3
- Walking shoud be the hard one. So the division is:
    - Hanqing: walk1_subject1 - walk2_subject4
    - Jak: walk3_subject1 - walk4_subject1
    - Sangwoo: run, jump and sprint

- Potential use:
    - aiming * 5(only with legs)
- Others:
    - With objects:
        - multipleActions 
    - rich contacts/movements:
        - dance
        - fallAndGetUp
        - fight
        - fightAndSports
        - ground
        - multipleActions 
        - push
        - pushAndFall
        - pushAndStumble
       
## Visualization
I used GMR for visualization because data has been retargeted on G1, more **real and direct** to observe. You could record **video** to better decide.

I add two features to the original code. replace `bvh_to_robot.py` and `robot_motion_viewer.py` to use.

Features: 
1. Press 'space' to pause
2. Print current frame at terminal

Tips:
1. For mac user, use mjpython like this command 

`mjpython scripts/bvh_to_robot.py --bvh_file 'your_path' --robot unitree_g1`

## Objects

- **Coarse clipping**: filter all the abnormal movements like limping.
    - **Naming**: original name + _seg i.bvh \
    for example: `'jumps1_subject1_seg1.bvh'`.
    - Note: **0 - around 70 frame** is the transition starting from initial pose. I would say clipping it.
- **Refined clipping**: 
    - Divide each segment into individual actions like walk, turn, stop.
    - We can **take down frame and corresponding action information** without making the file. If you guys have additional time I would recommend doing it.
## Velocity label
My understanding: For $i$ th frames, take $x,y,z$ component of the root joint(Hip), namely the first three elements. For example, the global velocity of $x$ component in $i$ th frame: 

$$V_{xi} = \frac{x_{i+1}-x_{i}}{t}$$
Where $t$ is the time for each frame, specified in the bvh file.
The results should be **$(N-1)\times3 $** tensor. For the last velocity, we can pad $0$ velocity.

**A plot script** is also prefered to see if we need to smooth the data.