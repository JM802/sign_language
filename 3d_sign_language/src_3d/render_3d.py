import os
import sys
import torch
import numpy as np
import trimesh
import pickle
from tqdm import tqdm
import config_3d
import glob
from scipy.interpolate import interp1d

# 加载 MANO 面片 (Faces) 方案
def load_mano_faces():
    # MANO 权重路径
    right_pkl_path = config_3d.RIGHT_PKL_PATH
    if not os.path.exists(right_pkl_path):
        raise FileNotFoundError(f"❌ 找不到 MANO 权重文件: {right_pkl_path}")
    
    with open(right_pkl_path, 'rb') as f:
        mano_data = pickle.load(f, encoding='latin1')
    return mano_data['f']#加载face拓扑

def interpolate_vertices_sequence(verts, factor):
    """
    不再强行对齐目标长度，而是按比例 factor 放大自身帧数。
    维持原有的动作速度和生命周期。
    """
    if verts is None or len(verts) < 2:
        return verts # 太短就不插值，直接原样返回
    
    orig_len = len(verts)
    target_len = orig_len * factor # 比如 33帧 -> 66帧
    
    x_orig = np.linspace(0, 1, orig_len)
    x_new = np.linspace(0, 1, target_len)
    
    if orig_len >= 4:
        f = interp1d(x_orig, verts, axis=0, kind='cubic')
    else:
        f = interp1d(x_orig, verts, axis=0, kind='linear')     
    return f(x_new)

def export_glb_sequence(word_dir, output_folder):
    # 加载数据
    if not os.path.exists(word_dir):
        print(f"❌ 找不到数据文件: {word_dir}")
        return
    
    #优先双手同框；若无则单手渲染。
    path_r = os.path.join(word_dir, "data_R.npz")
    path_l = os.path.join(word_dir, "data_L.npz")

    # 数据存在性探测
    has_r = os.path.exists(path_r)
    has_l = os.path.exists(path_l)

    if not has_r and not has_l:
        print(f"❌ 错误：该目录下没有任何 3D 数据文件: {word_dir}")
        return
    
    # 加载数据
    data_r = np.load(path_r) if has_r else None
    data_l = np.load(path_l) if has_l else None

    #提取顶点和 Root 位移 
    verts_r_local = data_r['vertices'] if has_r else None # (N, 778, 3)
    root_r_trans = data_r.get('joints', None)[:, 0:1, :] if has_r else None # 提取右手手腕位移 (N, 1, 3)

    verts_l_local = data_l['vertices'] if has_l else None
    root_l_trans = data_l.get('joints', None)[:, 0:1, :] if has_l else None # 提取左手手腕位移
  
    # 直接在顶点坐标上叠加全局 Root 位移。
    verts_r_global = verts_r_local + root_r_trans if has_r else None
    verts_l_global = verts_l_local + root_l_trans if has_l else None

    # 各自独立插值，互不干涉
    factor = config_3d.FACTOR # 升频倍数，可以改为3 或 4 让动作更慢更细致
    v_r_smooth = interpolate_vertices_sequence(verts_r_global, factor) if has_r else None
    v_l_smooth = interpolate_vertices_sequence(verts_l_global, factor) if has_l else None

    # 获取插值后的各自真实长度
    len_r_smooth = len(v_r_smooth) if v_r_smooth is not None else 0
    len_l_smooth = len(v_l_smooth) if v_l_smooth is not None else 0

    # 总循环长度取最大值，保证最长的那只手能播完
    num_frames = max(len_r_smooth, len_l_smooth)

    is_l_flipped = bool(data_l['is_flipped']) if has_l else False

    faces_base = load_mano_faces()
    faces_left = faces_base[:, [0, 2, 1]]
    
    # 初始化视频写入器
    os.makedirs(output_folder, exist_ok=True)
    print(f"🚀 正在导出 3D 模型序列至: {output_folder},(总帧数: {num_frames})")

    #渲染循环
    for i in tqdm(range(num_frames)):
        combined_mesh = None

        # 处理右手
        if v_r_smooth is not None and i < len_r_smooth:
            # 右手通常是基准，直接使用原始 faces_base
            mesh_r = trimesh.Trimesh(vertices=verts_r_global[i], faces=faces_base, process=False)
            # 给右手赋予 天蓝色 (稍微带点反光质感)
            mesh_r.visual.face_colors=[135, 206, 250, 255]
            combined_mesh = mesh_r

        # 处理左手
        if v_l_smooth is not None and i < len_l_smooth:
            actual_faces = faces_left if is_l_flipped else faces_base
            mesh_l = trimesh.Trimesh(vertices=v_l_smooth[i], faces=actual_faces, process=False)
            mesh_l.visual.face_colors = [255, 127, 80, 255] # 珊瑚橙

            # 给左手赋予 珊瑚橙色
            mesh_l.visual.face_colors = [255, 127, 80, 255]
            
            # 合并网格
            if combined_mesh is None:
                combined_mesh = mesh_l
            else:
                combined_mesh = combined_mesh + mesh_l

        if combined_mesh is not None:
            out_filepath = os.path.join(output_folder, f"frame_{i:03d}.glb")
            combined_mesh.export(out_filepath)

    print(f"✅ 导出成功！共生成 {num_frames} 个 3D 模型文件。")


if __name__ == "__main__":
   # 自动查找前 10 个测试样本
    base_db = "/home/jm802/sign_language/result_3d/database_npz/"
    all_dirs = sorted(glob.glob(os.path.join(base_db, "*/*/")))
    
    for i, sample_path in enumerate(all_dirs[:10]):
        parts = sample_path.strip('/').split('/')
        word, sid = parts[-2], parts[-1]
        out_path = f"/home/jm802/sign_language/result_3d/glb_test/{word}_{sid}/"
        print(f"🚀 [{i+1}/10] 处理: {word}")
        export_glb_sequence(sample_path, out_path)