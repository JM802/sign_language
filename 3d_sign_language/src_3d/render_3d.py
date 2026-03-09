import os
import sys
import torch
import numpy as np
import trimesh
import pickle
from tqdm import tqdm
import config_3d


# 加载 MANO 面片 (Faces) 方案
def load_mano_faces():
    # MANO 权重路径
    right_pkl_path = config_3d.RIGHT_PKL_PATH
    if not os.path.exists(right_pkl_path):
        raise FileNotFoundError(f"❌ 找不到 MANO 权重文件: {right_pkl_path}")
    
    with open(right_pkl_path, 'rb') as f:
        mano_data = pickle.load(f, encoding='latin1')
    return mano_data['f']#加载face拓扑

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

    # 确定总帧数(优先右手)
    num_frames = len(verts_r_global) if has_r else len(verts_l_global)
    # 加载基础面片拓扑
    faces_base = load_mano_faces()
    
    # 初始化视频写入器
    os.makedirs(output_folder, exist_ok=True)
    print(f"🚀 正在导出 3D 模型序列至: {output_folder}")

    #渲染循环
    for i in tqdm(range(num_frames)):
        #旋转180度，手背朝向相机，符合手语录制者（MANO默认手掌朝向相机）
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [0, 1, 0])

        combined_mesh = None

        # 处理右手   
        if has_r:
            mesh_r = trimesh.Trimesh(vertices=verts_r_global[i], faces=faces_base, process=False)
            mesh_r.apply_transform(rot)
            combined_mesh = mesh_r

        # 处理左手 
        if has_l:
            faces_l = faces_base[:, [0, 2, 1]]#修正法向量
            mesh_l = trimesh.Trimesh(vertices=verts_l_global[i], faces=faces_l, process=False)
            mesh_l.apply_transform(rot)
            
            # 将左右手网格合并为一个整体对象
            if combined_mesh is None:
                combined_mesh = mesh_l
            else:
                combined_mesh = combined_mesh + mesh_l

        # 导出当前帧为 .glb 格式
        # 文件名如 frame_000.glb, frame_001.glb
        out_filepath = os.path.join(output_folder, f"frame_{i:03d}.glb")
        
        # 赋予一个基础灰白材质（以便在查看器中看清肌肉阴影）
        combined_mesh.visual.face_colors = [200, 200, 200, 255]
        
        combined_mesh.export(out_filepath)

    print(f"✅ 导出成功！共生成 {num_frames} 个 3D 模型文件。")

if __name__ == "__main__":
   # 这里传入文件夹路径
    target_dir = "/home/jm802/sign_language/result_3d/database_npz/1DOLLAR/0719792557216079-1 DOLLAR/"
    # 将后缀改为目录名
    out_folder = "/home/jm802/sign_language/result_3d/glb_models/1DOLLAR_test_sample/"
    
    export_glb_sequence(target_dir, out_folder)