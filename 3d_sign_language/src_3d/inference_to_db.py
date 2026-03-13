import torch
import numpy as np
import os
import json
from tqdm import tqdm
from PIL import Image
import logging
from torchvision import transforms
import sys
import config_3d
import datetime

# 1. 路径与环境配置
S2HAND_PATH = config_3d.S2HAND_PATH
sys.path.append(S2HAND_PATH)
sys.path.append(os.path.join(S2HAND_PATH, "examples")) 
sys.path.append(os.path.join(S2HAND_PATH, "utils"))

from examples.models_new import Model
from examples.train_utils import load_model

# 获取当前脚本所在目录的上一级目录（即 3d_sign_language 文件夹）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 将 3d_sign_language 加入搜索路径
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from src_3d import config_3d  

# 配置日志系统
def setup_logging():
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    # 以当前时间命名日志文件
    log_filename = datetime.datetime.now().strftime("inference_%Y%m%d_%H%M%S.log")
    log_path = os.path.join(log_dir, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout) # 同时输出到终端
        ]
    )
    return log_path

# 2. 模拟参数配置
class ConfigStub:
    def __init__(self):
        self.train_requires = ['joints', 'verts', 'heatmaps', 'lights']
        self.test_requires = ['joints', 'verts']
        self.regress_mode = 'mano'
        self.use_mean_shape = False
        self.use_2d_as_attention = False
        self.renderer_mode = 'NR' 
        self.texture_mode = 'surf'
        self.image_size = 224
        self.train_datasets = ['FreiHand']
        self.use_pose_regressor = False
        self.pretrain_model = config_3d.PERTAINED_MODEL
        self.pretrain_segmnet = self.pretrain_texture_model = self.pretrain_rgb2hm = None

# 3. 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def init_engine():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = ConfigStub()
    model = Model(filename_obj=None, args=args).to(device)
    model, _ = load_model(model, args)
    model.eval()
    return model, device

@torch.no_grad()
def run_batch_inference(model, device, folder_path,side,batch_size=32):
    img_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
    if not img_files: return None

    # 用于暂存每个 mini-batch 的结果
    all_joints, all_vertices, all_pose, all_shape = [], [], [], []

    for i in range(0, len(img_files), batch_size):
        batch_files = img_files[i : i + batch_size]
        
        # 打包当前块
        batch_imgs = []
        for f in batch_files:
            img_pil = Image.open(os.path.join(folder_path, f)).convert('RGB')
            if side == 'L':
                img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
            batch_imgs.append(transform(img_pil))
        
        imgs = torch.stack(batch_imgs).to(device)

        K = torch.eye(4).unsqueeze(0).repeat(len(batch_files), 1, 1).to(device)
        
        # 核心推理
        out = model.predict_singleview(imgs, None, K, 'test', ['joints', 'verts'], None, None)
        
        # 将结果拉回 CPU 并追加到列表中
        all_joints.append(out['joints'].cpu().numpy())
        all_vertices.append(out['vertices'].cpu().numpy())
        all_pose.append(out['pose'].cpu().numpy())
        all_shape.append(out['shape'].cpu().numpy())
        
        # 手动释放当前 batch 的 GPU 显存引用
        del imgs, K, out
        torch.cuda.empty_cache()
    
    joints = np.concatenate(all_joints, axis=0)     # (N, 21, 3)
    vertices = np.concatenate(all_vertices, axis=0) # (N, 778, 3)
    pose = np.concatenate(all_pose, axis=0)         # (N, 48)
    shape = np.concatenate(all_shape, axis=0)       # (N, 10)

    # 如果是左手，将推理出的伪右手 3D 坐标在 X 轴上翻转回左手
    if side == 'L':
        joints[:, :, 0] = -joints[:, :, 0]
        vertices[:, :, 0] = -vertices[:, :, 0]
        # 后续在 render_3d.py 中渲染左手时，必须使用 MANO_LEFT.pkl 的面片，或者对 MANO_RIGHT 的面片做 [0, 2, 1] 绕序反转。
    
    return {"joints": joints, "vertices": vertices, "pose": pose, "shape": shape, "camera": None}

def build_database():
    log_path = setup_logging()
    logging.info(f"🚀 开始全量 3D 数据库构建任务。日志文件: {log_path}")

    # 1. 初始化引擎与设备
    model, device = init_engine()
    
    # 2. 结果保存路径配置
    db_root = config_3d.DB_ROOT
    os.makedirs(db_root, exist_ok=True)

    # 3. 初始化计数器
    processed_count = 0
    error_count=0

    # 4. 获取词条列表
    if not os.path.exists(config_3d.HAND_CROP_DIR):
        logging.warning(f"❌ 错误：找不到源数据目录 {config_3d.HAND_CROP_DIR}")
        return

    gloss_list = sorted(os.listdir(config_3d.HAND_CROP_DIR))
    
    # 外层循环：遍历词条 (如 8HOUR, 1DOLLAR 等)
    for gloss in tqdm(gloss_list, desc="Building 3D Database"):

        gloss_path = os.path.join(config_3d.HAND_CROP_DIR, gloss)
        if not os.path.isdir(gloss_path): 
            continue

        # 中层循环：遍历该词条下的具体视频 ID 文件夹
        vid_folders = sorted(os.listdir(gloss_path))
        for vid_id in vid_folders:

            vid_path = os.path.join(gloss_path, vid_id)
            # 标记当前视频是否成功产生数据
            # 内层循环：处理左手(L)和右手(R)
            has_data_in_this_vid = False
            for side in ['R', 'L']:
                target_dir = os.path.join(vid_path, side)
                
                # 检查 R/L 文件夹及 meta.json 是否存在
                meta_path = os.path.join(target_dir, "meta.json")
                if not os.path.exists(target_dir) or not os.path.exists(meta_path):
                    continue
                
                try:
                    # 核心推理：获取 3D 坐标
                    data = run_batch_inference(model, device, target_dir, side)
                    
                    if data is not None:
                        # 创建存储目录：db_root/词条/视频ID/
                        save_dir = os.path.join(db_root, gloss, vid_id)
                        os.makedirs(save_dir, exist_ok=True)
                        
                        # 存储为压缩格式 .npz
                        save_path = os.path.join(save_dir, f"data_{side}.npz")

                        is_flipped_flag = True if side == 'L' else False

                        np.savez_compressed(
                            save_path,
                            joints=data['joints'],
                            pose=data['pose'],
                            shape=data['shape'],
                            vertices=data['vertices'],
                            is_flipped=is_flipped_flag
                        )
                        has_data_in_this_vid=True
                    else:
                        error_count+=1
                except Exception as e:
                    logging.warning(f"跳过视频 {gloss}/{vid_id}/{side}: {str(e)}")
                    error_count += 1
            
            # 成功处理一个视频 ID 后自增计数
            if has_data_in_this_vid:
                processed_count += 1

            # 只有当该视频 ID 下至少有一个手(L或R)成功处理后，才增加总计数
    logging.info(f"🎉 提取任务完成！成功: {processed_count}, 失败: {error_count}")

if __name__ == "__main__":
    build_database()