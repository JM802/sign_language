import os
import sys
import torch
import json
import numpy as np
import trimesh
import pickle
from tqdm import tqdm
import config_3d
import glob
from scipy.interpolate import interp1d

# 加载 MANO 面片 (Faces) 方案
def load_mano_faces():
    pkl_path = config_3d.RIGHT_PKL_PATH
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"❌ 找不到 MANO 权重文件: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        mano_data = pickle.load(f, encoding='latin1')
    return mano_data['f']# 返回 face 拓扑

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

# 从 meta.json 中恢复全局 3D 轨迹
def get_global_offset_from_meta(meta_path, num_frames, spatial_scale=0.15, depth_scale=0.2):
    """将 2D 的 BBox 像素位移转换为 3D 空间的世界坐标位移"""
    if not os.path.exists(meta_path):
        return np.zeros((num_frames, 3))
        
    with open(meta_path, 'r') as f:
        meta_data = json.load(f)
        
    offsets = np.zeros((num_frames, 3))

    # 提取第一帧的 scale 作为 3D 空间的基准深度 (用于对齐 Z=0)
    base_scale = meta_data[0]['scale']

    for meta in meta_data:
        idx = meta['frame_idx']
        if idx >= num_frames: continue

        # 光学反比关系
        current_scale = meta['scale']
            
        # 获取 BBox 中心点
        cx = meta['offset'][0] + meta['scale'] / 2
        cy = meta['offset'][1] + meta['scale'] / 2
        img_w, img_h = meta['original_size']
        
        # 将像素坐标归一化并映射到 3D 空间
        # (X 轴向右为正，Y 轴在 3D 里是向上为正，所以要翻转)
        tx = spatial_scale * (cx - img_w / 2) / current_scale
        ty = -spatial_scale * (cy - img_h / 2) / current_scale 
          
        # 根据 Z = Constant / scale 计算真实深度
        # 为了让第一帧从 Z=0 开始，我们计算相对倒数差
        # 如果 current_scale 变大 (靠近镜头), (1/base - 1/current) 为正，Z 轴向屏幕外凸出
        tz = depth_scale * (1.0 - (base_scale / (current_scale + 1e-6)))
        
        offsets[idx] = [tx, ty, tz]
        
    return offsets

def export_glb_sequence(word_dir, output_folder):
    # 加载数据
    if not os.path.exists(word_dir):
        print(f"❌ 找不到数据文件: {word_dir}")
        return
    
    # 自动解析 word 和 vid_id
    parts = word_dir.strip('/').split('/')
    word, vid_id = parts[-2], parts[-1]
    
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

    # 获取原始顶点和局部微调位移
    v_r_local = data_r['vertices'] + data_r['joints'][:, 0:1, :] if has_r else None
    v_l_local = data_l['vertices'] + data_l['joints'][:, 0:1, :] if has_l else None

    # 加载 meta.json，把真正的空间大位移加回来
    meta_path_r = os.path.join(config_3d.HAND_CROP_DIR, word, vid_id, "R", "meta.json")
    meta_path_l = os.path.join(config_3d.HAND_CROP_DIR, word, vid_id, "L", "meta.json")
    
    if has_r:
        global_offset_r = get_global_offset_from_meta(meta_path_r, len(v_r_local))
        v_r_global_orig = v_r_local + global_offset_r[:, None, :] # 广播加到所有顶点上
    else:
        v_r_global_orig = None
        
    if has_l:
        global_offset_l = get_global_offset_from_meta(meta_path_l, len(v_l_local))
        v_l_global_orig = v_l_local + global_offset_l[:, None, :]
    else:
        v_l_global_orig = None

    # 各自独立插值，互不干涉
    factor = config_3d.FACTOR # 升频倍数，可以改为3 或 4让动作更慢更细致
    v_r_smooth = interpolate_vertices_sequence(v_r_global_orig, factor) if has_r else None
    v_l_smooth = interpolate_vertices_sequence(v_l_global_orig, factor) if has_l else None

    # 获取插值后的各自真实长度
    len_r_smooth = len(v_r_smooth) if v_r_smooth is not None else 0
    len_l_smooth = len(v_l_smooth) if v_l_smooth is not None else 0

    # 总循环长度取最大值，保证最长的那只手能播完
    num_frames = max(len_r_smooth, len_l_smooth)

    # 核心逻辑对齐你的思路：全场只用右手面片！
    faces_base = load_mano_faces() 
    # 用 [0, 2, 1] 把右手的面片顺序翻转，这就是完美的左手面片！
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
            mesh_r = trimesh.Trimesh(vertices=v_r_smooth[i], faces=faces_base, process=False)
            # 给右手赋予天蓝色
            mesh_r.visual.face_colors=[135, 206, 250, 255]
            combined_mesh = mesh_r

        # 处理左手
        if v_l_smooth is not None and i < len_l_smooth:
            mesh_l = trimesh.Trimesh(vertices=v_l_smooth[i], faces=faces_left, process=False)
            mesh_l.visual.face_colors = [255, 127, 80, 255] # 珊瑚橙
            
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
        temp_dir = sample_path.rstrip('/')
        vid_id = os.path.basename(temp_dir)
        word = os.path.basename(os.path.dirname(temp_dir))
        
        out_path = f"/home/jm802/sign_language/result_3d/glb_test/{word}_{vid_id}/"
        print(f"📦 [{i+1}/10] 任务开始: {word}")
        export_glb_sequence(sample_path, out_path)