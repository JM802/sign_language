import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)


def _env_path(name, default):
	value = os.getenv(name, default)
	return os.path.normpath(value)

# 数据根目录
DATA_ROOT = _env_path("SIGN_DATA_ROOT", os.path.join(BASE_DIR, "data"))

# 原始视频目录
VIDEO_DIR = os.path.join(DATA_ROOT, "wlasl-complete", "videos")

SPLIT_JSON_PATH = os.path.join(DATA_ROOT, "wlasl-complete", "nslt_300.json")

# 输出目录：存放提取好的 .npy 文件
SAVE_NPY_DIR = os.path.join(DATA_ROOT, "processed_features_300")
PROCESSED_FEATURE_DIR = SAVE_NPY_DIR

# 数据划分文件
TRAIN_MAP_PATH = os.path.join(DATA_ROOT, "train_map_300.txt")
VAL_MAP_PATH = os.path.join(DATA_ROOT, "val_map_300.txt")
TEST_MAP_PATH = os.path.join(DATA_ROOT, "test_map_300.txt")

# 归一化参数
GLOBAL_MEAN_PATH = os.path.join(DATA_ROOT, "global_mean_300_double_vel.npy")
GLOBAL_STD_PATH = os.path.join(DATA_ROOT, "global_std_300_double_vel.npy")

# 结果目录
RESULT_DIR = _env_path("SIGN_RESULT_DIR", os.path.join(BASE_DIR, "result"))
MODEL_SAVE_PATH = os.path.join(RESULT_DIR, "checkpoints")
CHECKPOINT_DIR = MODEL_SAVE_PATH
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model_300.ckpt")
LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "last_model_300.ckpt")
TRAIN_LOG_PATH = os.path.join(RESULT_DIR, "train_metrics_300.csv")

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
DEVICE = "Ascend"    # 昇腾环境；无卡时 train.py 会自动回退到 CPU

os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(SAVE_NPY_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
