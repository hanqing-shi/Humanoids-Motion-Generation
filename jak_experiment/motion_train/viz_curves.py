
import json
from pathlib import Path
import matplotlib.pyplot as plt

def plot_log(log_path: str, out_png: str):
    epochs, vals = [], []
    with open(log_path, "r") as f:
        for line in f:
            j = json.loads(line)
            epochs.append(j["epoch"])
            vals.append(j["val_loss"])
    plt.figure()
    plt.plot(epochs, vals, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Training Curve")
    plt.grid(True)
    plt.savefig(out_png, bbox_inches="tight")
    print(f"Saved {out_png}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", default="curve.png")
    args = ap.parse_args()
    plot_log(args.log, args.out)
