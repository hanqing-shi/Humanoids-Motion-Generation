import json, os, numpy as np
from bvh import Bvh
from scipy.spatial.transform import Rotation as R

def euler_to_quat_wxyz(euler_deg, rot_order):
    order = ''.join(ax[0].lower() for ax in rot_order)  # e.g. ['Zrotation','Yrotation','Xrotation'] -> 'zyx'
    q_xyzw = R.from_euler(order, euler_deg, degrees=True).as_quat()
    return np.c_[q_xyzw[:,3], q_xyzw[:,0], q_xyzw[:,1], q_xyzw[:,2]]  # wxyz

def make_quat_continuous(q):
    out = q.copy()
    for t in range(1, len(out)):
        if np.dot(out[t], out[t-1]) < 0: out[t] = -out[t]
    return out

def extract_joint_angles_rad(bvh, T, mapping_json):
    urdf_order = mapping_json["urdf_joint_order"]
    mp = mapping_json["map"]
    out = np.zeros((T, len(urdf_order)))
    cache = {}
    for jname in urdf_order:
        bj = mp[jname]["bvh_joint"]; axis = mp[jname]["axis"]
        if bj not in cache:
            ch = bvh.joint_channels(bj)
            M  = np.array(bvh.frames_joint_channels(bj, ch), dtype=float)
            cache[bj] = (ch, M)
        ch, M = cache[bj]
        idx = ch.index(axis)  # raises if wrong; fix mapping JSON
        out[:, urdf_order.index(jname)] = np.deg2rad(M[:, idx])
    return urdf_order, out

if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument("--bvh", required=True)
    p.add_argument("--mapping", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--scale_to_m", type=float, default=1.0)  # set 0.01 if BVH is in cm
    args = p.parse_args()

    with open(args.bvh, "r", encoding="utf-8", errors="ignore") as f:
        bvh = Bvh(f.read().lstrip("\ufeff"))

    root = bvh.get_joints_names()[0]
    ch_root = bvh.joint_channels(root)                           # Xpos,Ypos,Zpos, Zrot, Yrot, Xrot (varies)
    M_root = np.array(bvh.frames_joint_channels(root, ch_root), dtype=float)
    pos_m  = M_root[:, :3] * args.scale_to_m                     # GLOBAL root xyz (meters)
    eul    = M_root[:, 3:]                                       # GLOBAL root euler
    q_wxyz = make_quat_continuous(euler_to_quat_wxyz(eul, ch_root[3:]))

    mapping_json = json.load(open(args.mapping, "r", encoding="utf-8"))
    urdf_order, joint_rad = extract_joint_angles_rad(bvh, len(M_root), mapping_json)

    cfg = np.hstack([pos_m, q_wxyz, joint_rad])                  # [xyz, wxyz, joints(rad)]
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    np.savetxt(args.out_csv, cfg, delimiter=',')
    print(f"Saved {args.out_csv} shape={cfg.shape} (7 + {len(urdf_order)})")
