# import os
# import pandas as pd

# # === 配置 ===
# folder = "./dataset/data_joint/walk"   # 修改成你的路径

# header = [
#     "root_x", "root_y", "root_z", 
#     "root_qx", "root_qy", "root_qz", "root_qw",
#     "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
#     "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
#     "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
#     "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
#     "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
#     "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
#     "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
#     "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
#     "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
# ]

# def add_header_to_csv(csv_path, header):
#     # 读取
#     df = pd.read_csv(csv_path, header=None)

#     # 列数检查
#     if len(df.columns) != len(header):
#         print(f"[❌] {csv_path}: column mismatch {len(df.columns)} != {len(header)}")
#         return

#     # 写入 header
#     df.columns = header
#     df.to_csv(csv_path, index=False)
#     print(f"[✅] updated: {csv_path}")

# def main():
#     for f in os.listdir(folder):
#         if f.endswith(".csv"):
#             add_header_to_csv(os.path.join(folder, f), header)

# if __name__ == "__main__":
#     main()

import os
import pandas as pd
import glob

folder = "./dataset/data_joint/walk"

def fix_header(csv_path):
    df = pd.read_csv(csv_path, header=None)

    # if first two rows are identical → drop second row
    if df.iloc[0].equals(df.iloc[1]):
        print(f"[FIX] {csv_path}: duplicated header removed.")
        df = df.drop(index=1)
    else:
        print(f"[OK]  {csv_path}: no duplicated header.")

    # reset index & write back
    df.to_csv(csv_path, index=False, header=False)


def main():
    files = glob.glob(os.path.join(folder, "**/*.csv"), recursive=True)
    for f in files:
        fix_header(f)

if __name__ == "__main__":
    main()
