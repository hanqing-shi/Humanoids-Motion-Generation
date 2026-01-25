import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class JointDataset(Dataset):
    # same format as LAFAN1 retargeted dataset
    DEFAULT_JOINT_NAMES = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", 
        "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint", 
        "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint", 
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint", 
        "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", 
        "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint", 
        "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", 
        "right_wrist_yaw_joint"
    ]

    DEFAULT_ROOT_STATE = [
        'vel_x', 'vel_y', 'root_z',
        'root_qx', 'root_qy', 'root_qz', 'root_qw'
    ]

    def __init__(self, 
                 data_dir, 
                 label_dir, 
                 seq_len, 
                 motions, 
                 stride,
                 scaler=None):
        """
        Args:
            data_dir (str): Path to folder containing feature CSVs.
            label_dir (str): Path to folder containing label CSVs.
            seq_len (int): Sequence length per sample.
            motions (list[str]): List of motion subfolders (e.g., ['walk', 'run']).
            stride (int): Step size between consecutive sequences.
            scaler (dict, optional): Dictionary containing 'min', 'max', 'range'. 
                                     - For Training: Leave None to compute from data.
                                     - For Inference/Validation: Must provide the training set's scaler.
        """
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.seq_len = seq_len
        self.stride = stride
        
        # read csv files
        self.selected_cols = self.DEFAULT_ROOT_STATE + self.DEFAULT_JOINT_NAMES
        self.input_files = []
        for motion in motions:
            motion_dir = os.path.join(data_dir, motion)
            csvs = sorted(glob.glob(os.path.join(motion_dir, "*.csv")))
            self.input_files.extend(csvs)

        if len(self.input_files) == 0:
            raise FileNotFoundError(f"No CSV input files found in {motions} under {data_dir}")
        
        print(f"✅ Found {len(self.input_files)} CSV files from motions {motions}")

        # get motion sequences with sliding window
        self.samples = [] 
        all_data_list = []
        
        for file_path in self.input_files:
            df = pd.read_csv(file_path, usecols=self.selected_cols)
            T = len(df)
            
            # compute global scaler stats if not provided
            if scaler is None:
                feat = df[self.selected_cols].values.astype(np.float32)
                all_data_list.append(feat)

            if T < seq_len:
                continue
            
            # Create sliding window indices
            starts = list(range(0, T - seq_len + 1, stride))
            for s in starts:
                self.samples.append((file_path, s))

        # min-max normalization
        if scaler is not None:
            self.min = scaler['min'].cpu()
            self.max = scaler['max'].cpu()
            self.range = scaler['range'].cpu()
        else:
            if not all_data_list:
                raise ValueError("No data loaded to compute scaler stats!")
            
            all_data = np.concatenate(all_data_list, axis=0)
            self.min = torch.tensor(all_data.min(axis=0), dtype=torch.float32)
            self.max = torch.tensor(all_data.max(axis=0), dtype=torch.float32)
            
            # Compute range and handle division by zero
            self.range = self.max - self.min
            self.range[self.range < 1e-4] = 1.0 

        print(f"📊 Total {len(self.samples)} subsequences.")

        # Avoid normalizing quaternion columns
        self.norm_mask = torch.ones(len(self.selected_cols), dtype=torch.bool)
        for i, col_name in enumerate(self.selected_cols):
            if 'root_q' in col_name: 
                self.norm_mask[i] = False

    def get_scaler(self):
        """
        Returns the scaler statistics. 
        """
        return {
            'min': self.min.clone().cpu(),
            'max': self.max.clone().cpu(),
            'range': self.range.clone().cpu()
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, start = self.samples[idx]

        rel_path = os.path.relpath(file_path, self.data_dir)
        stem, ext = os.path.splitext(rel_path)
        
        if '_feature' in stem:
            label_rel = stem.replace('_feature', '_label') + ext
        else:
            label_rel = stem + "_label" + ext
            
        label_path = os.path.join(self.label_dir, label_rel)

        df_data = pd.read_csv(file_path, usecols=self.selected_cols) 
        data = df_data[self.selected_cols].values.astype(np.float32)
        label = pd.read_csv(label_path).values.astype(np.float32)

        # sclicing
        data_seq = data[start : start + self.seq_len]
        label_seq = label[start : start + self.seq_len]

        data_seq = torch.tensor(data_seq, dtype=torch.float32)
        label_seq = torch.tensor(label_seq, dtype=torch.float32)

        mask = self.norm_mask
        
        # min-max normalization to [-1, 1]
        data_seq[..., mask] = 2 * (data_seq[..., mask] - self.min[mask]) / self.range[mask] - 1

        return data_seq, label_seq