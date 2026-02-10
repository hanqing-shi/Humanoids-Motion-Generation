import torch

ckpt_path = "checkpoints/TrajCVAE_best_walk.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")

print("Checkpoint type:", type(ckpt))

if isinstance(ckpt, dict):
    print("Top-level keys:", ckpt.keys())

    for k in ["state_dict", "model", "model_state_dict"]:
        if k in ckpt:
            print(f"\nUsing nested key: {k}")
            sd = ckpt[k]
            break
    else:
        sd = ckpt
else:
    sd = ckpt

print("\nNumber of tensors:", len(sd))
print("\nSample parameter shapes:")
for name, tensor in list(sd.items())[:10]:
    print(f"{name:40s} {tuple(tensor.shape)}")
