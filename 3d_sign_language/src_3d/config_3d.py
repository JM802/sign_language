"""
config_3d.py — 三维手部重建流水线配置
阶段：Stage 1 手部裁剪预处理 (Hand Cropping for S2HAND)
"""
import os

# ========================= 数据路径 =========================
DATA_ROOT = "/home/jm802/sign_language/data"

# 原始视频目录（按 video_id 命名的 .mp4 文件）
VIDEO_DIR = os.path.join(DATA_ROOT, "ASL_Citizen", "videos")

# 更新后的 CSV 路径（先测试 train.csv）
CSV_PATH = os.path.join(DATA_ROOT, "ASL_Citizen", "splits", "train.csv")

# ========================= 输出路径 =========================
# 裁剪后的手部图片根目录
# 结构：HAND_CROP_DIR/<gloss_id>/<split>/<video_id>/frame_0000.jpg
HAND_CROP_DIR = os.path.join(DATA_ROOT, "hand_crops_224")

# BBox 缓存（每个视频每帧的检测框，.npy，方便调试/复用）
BBOX_CACHE_DIR = os.path.join(DATA_ROOT, "bbox_cache")

# 处理日志目录（同脚本文件夹下）
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

# ========================= 图像参数 =========================
IMAGE_SIZE      = 224     # S2HAND 标准输入 224×224
BBOX_EXPAND_RATIO = 1.45  # BBox 扩展系数（1.2~1.5 均可）


# ========================= MediaPipe 参数 =========================
MP_DETECTION_CONFIDENCE = 0.5
MP_TRACKING_CONFIDENCE  = 0.5
MP_MAX_NUM_HANDS        = 2   # 手语通常需要双手

# ========================= 并行 / 保存参数 =========================
NUM_WORKERS  = 4    # 多进程并行（0 = 自动=CPU核心数）
JPEG_QUALITY = 95   # JPEG 保存质量

# ========================= 跳过控制 =========================
# True  → 已处理过的视频直接跳过（断点续跑）
# False → 强制重新处理所有视频
SKIP_EXISTING = True

#高斯平滑
SIGMA=1.7

#debug目录
DEBUG_OUT_DIR="/home/jm802/sign_language/debug_results"

#S2HAND路径配置
S2HAND_SDK_PATH=S2HAND_PATH = "/home/jm802/sign_language/S2HAND_code"
PERTAINED_MODEL="/home/jm802/sign_language/S2HAND_code/checkpoints/checkpoint.pth"
DB_ROOT="/home/jm802/sign_language/result_3d/database_npz"
RIGHT_PKL_PATH="/home/jm802/sign_language/3d_sign_language/mano_v1_2/models/MANO_RIGHT.pkl"

FACTOR=2
