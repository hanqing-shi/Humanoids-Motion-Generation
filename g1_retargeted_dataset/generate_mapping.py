import re, json, numpy as np
from bvh import Bvh
import pinocchio as pin

def urdf_introspect(urdf_path):
    model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
    # skip 0:universe, 1:freeflyer
    urdf_joint_names = [model.names[j] for j in range(2, model.njoints)]
    return urdf_joint_names

def bvh_introspect(bvh_path):
    with open(bvh_path, "r", encoding="utf-8", errors="ignore") as f:
        bvh = Bvh(f.read().lstrip("\ufeff"))
    joint_names = bvh.get_joints_names()
    channels = {jn: bvh.joint_channels(jn) for jn in joint_names}
    return joint_names, channels

def norm(s): return re.sub(r'[^a-z0-9]+','',s.lower())

def propose_mapping(urdf_names, bvh_names, bvh_channels):
    nmap = {norm(n): n for n in bvh_names}
    out = {}
    for uj in urdf_names:
        k = norm(uj)
        # guess BVH joint by fuzzy name
        guess = nmap.get(k, None)
        if not guess:
            cands = [n for nk,n in nmap.items() if k in nk or nk in k]
            guess = (cands[0] if cands else bvh_names[0])
        ch = bvh_channels.get(guess, [])
        # prefer Z, then Y, then X if present
        axis = next((a for a in ["Zrotation","Yrotation","Xrotation"] if a in ch), None)
        out[uj] = {"bvh_joint": guess, "axis": axis}
    return out

if __name__ == "__main__":
    import argparse, os
    p = argparse.ArgumentParser()
    p.add_argument("--urdf", required=True)
    p.add_argument("--bvh", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    urdf_names = urdf_introspect(args.urdf)
    bvh_names, bvh_ch = bvh_introspect(args.bvh)
    mapping = propose_mapping(urdf_names, bvh_names, bvh_ch)

    payload = {
        "urdf_joint_order": urdf_names,
        "map": mapping,
        "notes": "Review/fix any wrong entries. 'axis' must exist in that BVH joint's channels."
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("Wrote mapping template:", args.out)
