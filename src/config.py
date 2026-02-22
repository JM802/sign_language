import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据根目录
DATA_ROOT = "/home/jm802/sign_language/data"

# 原始视频目录
VIDEO_DIR = os.path.join(DATA_ROOT, "wlasl-complete", "videos")

# 🔥 核心修改：直接使用官方提供的 300 词划分文件
# 这个文件里已经分好了 train/val/test，不需要我们自己分
SPLIT_JSON_PATH = os.path.join(DATA_ROOT, "wlasl-complete", "nslt_300.json")

# 输出目录：存放提取好的 .npy 文件
# 建议单独建一个文件夹，和原始数据分开
SAVE_NPY_DIR = os.path.join(DATA_ROOT, "processed_features_300")

# 结果目录
RESULT_DIR = "/home/jm802/sign_language/result"
MODEL_SAVE_PATH = os.path.join(RESULT_DIR, "checkpoints")

# ================= 数据参数 =================
# MediaPipe特征维度计算:
# Pose(只取上半身0-24点=25个) * 2(x,y) = 50
# Left Hand(21个) * 2(x,y) = 42
# Right Hand(21个) * 2(x,y) = 42
# 加速度 Δx, Δy
#268维
INPUT_SIZE = 268    

SEQ_LEN = 64         # 序列统一长度
NUM_CLASSES = 300    # 类别数

# ================= 训练参数 =================
BATCH_SIZE = 64
EPOCHS = 80
LEARNING_RATE = 1e-3
DEVICE = "cuda"      # 你的环境有GPU，务必用cuda