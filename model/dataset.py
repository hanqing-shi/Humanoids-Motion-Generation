import os, glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class MotionDataset(Dataset):
    def __init__(self, 
                 data_dir, 
                 label_dir, 
                 seq_len=300, 
                 motions=['walk'],  # list of motion subfolders
                 columns=("pos", "ori", "vel", "angvel"),
                 stride=100,
                 transform=None):
        """
        Args:
            data_dir (str): path to folder containing feature CSVs
            label_dir (str): path to folder containing label CSVs
            seq_len (int): sequence length per sample
            motions (list[str]): list of motion subfolders
            columns (tuple[str]): which feature groups to include, e.g. ("pos", "ori")
            stride (int): step between consecutive sequences from the same file
            transform (callable): optional preprocessing function
        """
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.seq_len = seq_len
        self.columns = columns
        self.stride = stride
        self.transform = transform
        
        # --------------------------------------------------
        # Collect all feature CSV files
        # --------------------------------------------------
        self.input_files = []
        for motion in motions:
            motion_dir = os.path.join(data_dir, motion)
            csvs = sorted(glob.glob(os.path.join(motion_dir, "*.csv")))
            self.input_files.extend(csvs)

        if len(self.input_files) == 0:
            raise FileNotFoundError(f"No CSV input files found in {motions} under {data_dir}")

        print(f"✅ Found {len(self.input_files)} CSV files from motions {motions}")

        # --------------------------------------------------
        # Define feature columns
        # --------------------------------------------------
        frame_names = ['pelvis', 'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link', 'left_knee_link', 
                       'left_ankle_pitch_link', 'left_ankle_roll_link', 'pelvis_contour_link', 'right_hip_pitch_link', 
                       'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_pitch_link', 
                       'right_ankle_roll_link', 'waist_yaw_link', 'waist_roll_link', 'torso_link', 'head_link', 
                       'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 
                       'left_elbow_link', 'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link', 
                       'left_rubber_hand', 'logo_link', 'right_shoulder_pitch_link', 'right_shoulder_roll_link', 
                       'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_roll_link', 'right_wrist_pitch_link', 
                       'right_wrist_yaw_link', 'right_rubber_hand']
        
        self.root_state = [
            'root_x', 'root_y', 'root_z',
            'root_qw', 'root_qx', 'root_qy', 'root_qz',
            'root_vx', 'root_vy', 'root_vz',
            'root_wx', 'root_wy', 'root_wz'
        ]
        
        self.body_state = []
        for f in frame_names:
            self.body_state += [
                f'{f}_pos_x', f'{f}_pos_y', f'{f}_pos_z',
                f'{f}_ori_w', f'{f}_ori_x', f'{f}_ori_y', f'{f}_ori_z',
                f'{f}_vel_x', f'{f}_vel_y', f'{f}_vel_z',
                f'{f}_angvel_x', f'{f}_angvel_y', f'{f}_angvel_z'
            ]
        
        self.col_dict = {
            "pos":   ["pos_x", "pos_y", "pos_z"],
            "ori":   ["ori_w", "ori_x", "ori_y", "ori_z"],
            "vel":   ["vel_x", "vel_y", "vel_z"],
            "angvel": ["angvel_x", "angvel_y", "angvel_z"],
        }

        # filter columns by user-specified groups
        self.selected_body_cols = [
            name for name in self.body_state
            if any(suffix in name for c in columns for suffix in self.col_dict[c])
        ]
        self.selected_cols = self.root_state + self.selected_body_cols

        # --------------------------------------------------
        # Preprocess: compute slicing points for each file
        # --------------------------------------------------
        self.samples = []  # [(file_path, start_idx)]

        for file_path in self.input_files:
            df = pd.read_csv(file_path)
            T = len(df)
            if T < seq_len:
                continue  # skip too short
            starts = list(range(0, T - seq_len + 1, stride))
            for s in starts:
                self.samples.append((file_path, s))
        
        print(f"📊 Total {len(self.samples)} subsequences from {len(self.input_files)} files")

    # --------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # --------------------------------------------------
    def __getitem__(self, idx):
        file_path, start = self.samples[idx]

        # get matching label file (replace _feature -> _label)
        rel_path = os.path.relpath(file_path, self.data_dir)
        label_path = os.path.join(self.label_dir, rel_path.replace('_feature', '_label'))

        # read data
        data = pd.read_csv(file_path)[self.selected_cols].values.astype(np.float32)
        label = pd.read_csv(label_path).values.astype(np.float32)

        assert len(label) == len(data), f"Length mismatch: {file_path}"

        # extract segment
        data_seq = data[start : start + self.seq_len]
        label_seq = label[start : start + self.seq_len]

        data_seq = torch.tensor(data_seq, dtype=torch.float32)
        label_seq = torch.tensor(label_seq, dtype=torch.float32)

        if self.transform:
            data_seq = self.transform(data_seq)

        return data_seq, label_seq
