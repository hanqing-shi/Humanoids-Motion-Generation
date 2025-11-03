# dataset.py
import os, glob, json, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset

def _paired_training_path(modelstate_path: str) -> str:
    base = os.path.basename(modelstate_path).replace("_modelstate.csv", "")
    return os.path.join(os.path.dirname(modelstate_path), f"{base}_training.csv")

class MotionStateDataset(Dataset):
    """
    Yields:
      cond  : [T, 6]  -> root_vx,vy,vz, root_omx,omy,omz
      target: [T, D]  -> modelstate (root pose + v/ω + joints)
      body  : [T, 3B] -> body_pos_local from *_training.csv (auxiliary target)
      start : int
    """
    def __init__(self, data_dir, split="train", seq_len=60, stride=2,
                 use_cond=True, use_body_aux=True, stats_path=None, body_stats_path=None):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*_modelstate.csv")))
        if len(self.files)==0:
            raise FileNotFoundError(f"No *_modelstate.csv in {data_dir}")
        self.seq_len=seq_len; self.stride=stride
        self.use_cond=use_cond; self.use_body_aux=use_body_aux

        # ---------- core normalization (modelstate) ----------
        if stats_path and os.path.exists(stats_path):
            with open(stats_path) as f: stats=json.load(f)
            self.mean=np.array(stats["mean"]); self.std=np.array(stats["std"])
        else:
            all_data=[]
            for f in self.files:
                arr=pd.read_csv(f).drop(columns=["frame"]).to_numpy()
                all_data.append(arr)
            all_data=np.concatenate(all_data,0)
            self.mean=all_data.mean(0); self.std=all_data.std(0)+1e-8
            if split=="train":
                with open(os.path.join(data_dir,"norm_stats.json"),"w") as f:
                    json.dump({"mean":self.mean.tolist(),"std":self.std.tolist()},f)

        # ---------- body aux normalization (from *_training.csv) ----------
        self.body_mean=None; self.body_std=None
        if self.use_body_aux:
            if body_stats_path and os.path.exists(body_stats_path):
                with open(body_stats_path) as f: bstats=json.load(f)
                self.body_mean=np.array(bstats["mean"]); self.body_std=np.array(bstats["std"])
            else:
                all_body=[]
                for f in self.files:
                    tpath=_paired_training_path(f)
                    if not os.path.exists(tpath): continue
                    A=pd.read_csv(tpath, header=None).to_numpy(dtype=float)
                    # infer counts
                    D_model = pd.read_csv(f).drop(columns=["frame"]).shape[1]
                    J = D_model - 13
                    D_train = A.shape[1]
                    N = int((D_train - 13 - J) // 13)
                    body_pos = A[:, 13:13+3*N]
                    all_body.append(body_pos)
                if len(all_body):
                    all_body=np.concatenate(all_body,0)
                    self.body_mean=all_body.mean(0); self.body_std=all_body.std(0)+1e-8
                if split=="train" and self.body_mean is not None:
                    with open(os.path.join(data_dir,"body_norm_stats.json"),"w") as f:
                        json.dump({"mean":self.body_mean.tolist(),"std":self.body_std.tolist()},f)

        # ---------- index samples ----------
        self.samples=[]
        for f in self.files:
            arr=pd.read_csv(f).drop(columns=["frame"]).to_numpy()
            for i in range(0,len(arr)-seq_len,stride):
                self.samples.append((f,i))
        print(f"[{split}] Loaded {len(self.samples)} sequences from {len(self.files)} files")

    def __len__(self): return len(self.samples)

    def __getitem__(self,idx):
        path,start=self.samples[idx]
        arr=pd.read_csv(path).drop(columns=["frame"]).to_numpy()
        # normalize core
        arr=(arr-self.mean)/self.std
        seq=arr[start:start+self.seq_len]
        seq=torch.from_numpy(seq).float()

        cond=None
        if self.use_cond:
            # FIX: root linear + angular velocity = cols 7..12
            cond=seq[:,7:13]  # [vx,vy,vz, omx,omy,omz]

        body=None
        if self.use_body_aux:
            tpath=_paired_training_path(path)
            if os.path.exists(tpath) and self.body_mean is not None:
                A=pd.read_csv(tpath, header=None).to_numpy(dtype=float)
                # infer body count N from dims
                D_model = pd.read_csv(path).drop(columns=["frame"]).shape[1]
                J = D_model - 13
                D_train = A.shape[1]
                N = int((D_train - 13 - J) // 13)
                body_pos = A[:, 13:13+3*N]    # [T, 3N] local
                # slice same window + normalize
                body_pos = (body_pos - self.body_mean) / self.body_std
                body = torch.from_numpy(body_pos[start:start+self.seq_len]).float()

        return cond, seq, body, start
