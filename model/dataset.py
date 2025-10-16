import os, glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class MotionDataset(Dataset):
    def __init__(self, 
                 data_dir, 
                 label_dir, 
                 seq_len=30, 
                 motions = ['walk','run'], 
                 columns=("pos", "ori"),
                 transform=None):
        """
        Args:
            data_dir (str): path to folder containing CSVs
            label_dir (str): path to folder cotaning Velocity labels
            seq_len (int): sequence length per sample
            columns (tuple[str]): which feature groups to include, e.g. ("pos", "ori")
            transform (callable): preprocessing mirror transform
        """
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.seq_len = seq_len
        self.columns = columns
        self.transform = transform
        
        # Adding csv files
        self.input_files = []
        for motion in motions:
            motion_dir = os.path.join(data_dir,motion)
            csvs = sorted(glob.glob(os.path.join(motion_dir, "*.csv")))
            self.input_files.extend(csvs)

        if len(self.input_files) == 0:
            raise FileNotFoundError(f"No CSV input_files found in {motions} under {data_dir}")

        print(f"Found {len(self.input_files)} input_files from motions {motions}")


        # TODO: define exact name
        self.col_dict = {
            "pos":   ["pos_x", "pos_y", "pos_z"],
            "ori":   ["quat_w", "quat_x", "quat_y", "quat_z"],
            "vel":   ["vel_x", "vel_y", "vel_z"],
            "acc":   ["acc_x", "acc_y", "acc_z"],
            "joint": [f"joint_{i}" for i in range(1, 29)],
        }

        self.cols = sum([self.col_dict[c] for c in columns], [])

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        # get label path
        input_path = self.input_files[idx]
        rel_path = os.path.relpath(input_path,self.data_dir)
        label_path = os.path.join(self.label_dir, rel_path)

        # read data 
        data = pd.read_csv(input_path)[self.cols].values.astype(np.float32)
        label = pd.read_csv(label_path).values.astype(np.float32)

        T, D = data.shape
        assert len(label) == T, f"Length mismatch: {input_path}"

        if len(data) < self.seq_len:
            raise ValueError(f"{self.input_files[idx]} too short for seq_len={self.seq_len}")
        
        # sampling
        start = np.random.randint(0, len(data) - self.seq_len)
        data_seq = torch.tensor(data[start:start+self.seq_len], dtype=torch.float32)
        label_seq = torch.tensor(label[start:start+self.seq_len], dtype=torch.float32)
        if self.transform:
            seq = self.transform(seq)

        return data_seq, label_seq

