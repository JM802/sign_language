import os
import sys
import torch
import numpy as np
import trimesh
import pickle
from tqdm import tqdm
import config_3d
import glob


def load_mano_faces():
    right_pkl_path = config_3d.RIGHT_PKL_PATH
    if not os.path.exists(right_pkl_path):
        raise FileNotFoundError(f"❌ 找不到 MANO 权重文件: {right_pkl_path}")
    with open(right_pkl_path, 'rb') as f:
        mano_data = pickle.load(f, encoding='latin1')
    return mano_data['f']


def check_normal_direction(vertices, faces, label=""):
    """
    验证法向量方向是否朝外。
    返回: (volume, outward_ratio, is_ok)
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
    # 1. 体积符号检测（正 = 法向量朝外）
    vol = mesh.volume
    
    # 2. 法向量朝外比例检测
    center = vertices.mean(axis=0)
    face_centers = vertices[faces].mean(axis=1)         # (F, 3)
    face_normals = mesh.face_normals                     # (F, 3)
    outward_vec = face_centers - center                  # 从重心指向面中心
    dot = (face_normals * outward_vec).sum(axis=1)       # 点积
    outward_ratio = (dot > 0).mean()
    
    is_ok = vol > 0 and outward_ratio > 0.5
    status = "✅" if is_ok else "❌"
    
    if label:
        print(f"  {status} [{label}] vol={vol:+.4f}  朝外比例={outward_ratio:.1%}  {'正常' if is_ok else '法向量反转！'}")
    
    return vol, outward_ratio, is_ok


def export_glb_sequence(word_dir, output_folder, validate=True):
    if not os.path.exists(word_dir):
        print(f"❌ 找不到数据文件: {word_dir}")
        return None

    path_r = os.path.join(word_dir, "data_R.npz")
    path_l = os.path.join(word_dir, "data_L.npz")
    has_r = os.path.exists(path_r)
    has_l = os.path.exists(path_l)

    if not has_r and not has_l:
        print(f"❌ 该目录下没有任何 3D 数据文件: {word_dir}")
        return None

    data_r = np.load(path_r) if has_r else None
    data_l = np.load(path_l) if has_l else None

    verts_r_local = data_r['vertices'] if has_r else None
    root_r_trans  = data_r['joints'][:, 0:1, :] if has_r else None
    verts_l_local = data_l['vertices'] if has_l else None
    root_l_trans  = data_l['joints'][:, 0:1, :] if has_l else None

    verts_r_global = verts_r_local + root_r_trans if has_r else None
    verts_l_global = verts_l_local + root_l_trans if has_l else None

    len_r = len(verts_r_global) if has_r else 0
    len_l = len(verts_l_global) if has_l else 0
    num_frames = max(len_r, len_l)

    is_l_flipped = bool(data_l['is_flipped']) if has_l else False

    faces_base = load_mano_faces()
    faces_left  = faces_base[:, [0, 2, 1]]

    os.makedirs(output_folder, exist_ok=True)
    print(f"\n🚀 导出至: {output_folder}  (帧数={num_frames}, L翻转={is_l_flipped})")

    # ── 验证统计 ──────────────────────────────────────────
    stats = {
        'R': {'ok': 0, 'bad': 0, 'vols': [], 'ratios': []},
        'L': {'ok': 0, 'bad': 0, 'vols': [], 'ratios': []},
    }

    for i in tqdm(range(num_frames), desc="渲染帧"):
        combined_mesh = None

        # 右手
        if has_r and i < len_r:
            v_r = verts_r_global[i]
            mesh_r = trimesh.Trimesh(vertices=v_r, faces=faces_base, process=False)
            combined_mesh = mesh_r

            if validate:
                vol, ratio, ok = check_normal_direction(v_r, faces_base, f"R frame{i:03d}")
                stats['R']['vols'].append(vol)
                stats['R']['ratios'].append(ratio)
                stats['R']['ok' if ok else 'bad'] += 1

        # 左手
        if has_l and i < len_l:
            v_l = verts_l_global[i]
            actual_faces = faces_left if is_l_flipped else faces_base
            mesh_l = trimesh.Trimesh(vertices=v_l, faces=actual_faces, process=False)
            combined_mesh = combined_mesh + mesh_l if combined_mesh else mesh_l

            if validate:
                vol, ratio, ok = check_normal_direction(v_l, actual_faces, f"L frame{i:03d}")
                stats['L']['vols'].append(vol)
                stats['L']['ratios'].append(ratio)
                stats['L']['ok' if ok else 'bad'] += 1

        if combined_mesh is not None:
            combined_mesh.visual.face_colors = [200, 200, 200, 255]
            combined_mesh.export(os.path.join(output_folder, f"frame_{i:03d}.glb"))

    # ── 打印汇总 ──────────────────────────────────────────
    if validate:
        print("\n── 法向量验证汇总 ──────────────────────────────")
        for side in ['R', 'L']:
            s = stats[side]
            if s['ok'] + s['bad'] == 0:
                continue
            total = s['ok'] + s['bad']
            avg_vol   = np.mean(s['vols'])
            avg_ratio = np.mean(s['ratios'])
            bad_frames = [i for i, v in enumerate(s['vols']) if v < 0]
            print(f"  {'右手' if side=='R' else '左手'}: {s['ok']}/{total} 帧正常  "
                  f"平均体积={avg_vol:+.4f}  平均朝外比={avg_ratio:.1%}")
            if bad_frames:
                print(f"    ⚠️  异常帧索引: {bad_frames}")
        print("────────────────────────────────────────────────\n")

    return stats


if __name__ == "__main__":
    base_db  = "/home/jm802/sign_language/result_3d/database_npz/"
    all_dirs = sorted(glob.glob(os.path.join(base_db, "*/*/")))

    all_stats = []
    for i, sample_path in enumerate(all_dirs[:10]):
        parts = sample_path.strip('/').split('/')
        word, sid = parts[-2], parts[-1]
        out_path = f"/home/jm802/sign_language/result_3d/glb_test/{word}_{sid}/"
        print(f"\n{'='*55}")
        print(f"🚀 [{i+1}/10] 词条: {word}  ID: {sid}")
        stats = export_glb_sequence(sample_path, out_path, validate=True)
        if stats:
            all_stats.append((word, sid, stats))

    # ── 10个样本总体报告 ─────────────────────────────────
    print(f"\n{'='*55}")
    print("📊 10 个测试样本总体报告")
    print(f"{'='*55}")
    for word, sid, stats in all_stats:
        r_ok  = stats['R']['ok'];  r_bad  = stats['R']['bad']
        l_ok  = stats['L']['ok'];  l_bad  = stats['L']['bad']
        r_str = f"R {r_ok}/{r_ok+r_bad}" if r_ok+r_bad > 0 else "R  N/A"
        l_str = f"L {l_ok}/{l_ok+l_bad}" if l_ok+l_bad > 0 else "L  N/A"
        flag  = "✅" if r_bad == 0 and l_bad == 0 else "❌"
        print(f"  {flag} {word}/{sid}  {r_str}  {l_str}")
    print(f"{'='*55}")