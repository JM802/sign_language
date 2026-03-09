import os
import json
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import config  # 确保同目录下有 config.py

# =============================
# 初始化 MediaPipe Holistic
# =============================
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True
)

# =============================
# 提取视频片段特征（指定帧区间）
# =============================
def extract_features(video_path, start_frame, end_frame):
    cap = cv2.VideoCapture(video_path)
    frames_data = []
    current_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_frame += 1

        # 跳过前段
        if current_frame < start_frame:
            continue

        # 超过后段退出
        if current_frame > end_frame:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        row = []

        # Pose (25 点 -> 50维)
        if results.pose_landmarks:
            for i in range(25):
                lm = results.pose_landmarks.landmark[i]
                row.extend([lm.x, lm.y])
        else:
            row.extend([0.0] * 50)

        # Left Hand (21 点 -> 42维)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                row.extend([lm.x, lm.y])
        else:
            row.extend([0.0] * 42)

        # Right Hand (21 点 -> 42维)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                row.extend([lm.x, lm.y])
        else:
            row.extend([0.0] * 42)

        frames_data.append(row)

    cap.release()
    return np.array(frames_data, dtype=np.float32)


# =============================
# 计算全局均值与标准差（仅训练集）
# =============================
def calculate_global_stats(train_list_lines):
    print("🧮 正在计算全局均值和标准差...")

    all_data = []

    # 解析 train_list_lines，每一行是 "path,label"
    for line in tqdm(train_list_lines, desc="Loading Train Data"):
        npy_path = line.split(',')[0] # 获取路径
        
        if os.path.exists(npy_path):
            arr = np.load(npy_path)
            if len(arr) > 0:
                all_data.append(arr)

    if not all_data:
        print("❌ 错误：没有加载到训练数据！")
        return None, None

    # 拼接并计算(做正态标准化)
    concatenated = np.concatenate(all_data, axis=0)
    mean = np.mean(concatenated, axis=0)
    std = np.std(concatenated, axis=0)
    
    # 防止除以0
    std = np.where(std == 0, 1.0, std)

    print(f"✅ 统计完成。Mean shape: {mean.shape}, Std shape: {std.shape}")
    return mean, std


# =============================
# 主流程
# =============================
def process_dataset():
    # 检查配置
    if not os.path.exists(config.SPLIT_JSON_PATH):
        print(f"❌ 错误：找不到 JSON 文件 {config.SPLIT_JSON_PATH}")
        return

    # 创建目录
    os.makedirs(config.SAVE_NPY_DIR, exist_ok=True)
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)

    print(f"📖 读取划分文件: {config.SPLIT_JSON_PATH}")
    with open(config.SPLIT_JSON_PATH, "r") as f:
        split_data = json.load(f)

    subsets = {"train": [], "val": [], "test": []}

    processed_count = 0
    missing_count = 0

    print("🚀 开始提取特征 (支持断点续传)...")

    for video_id, info in tqdm(split_data.items()):
        subset = info["subset"]
        label = info["action"][0]
        start_frame = info["action"][1]
        end_frame = info["action"][2]

        vid_path = os.path.join(config.VIDEO_DIR, f"{video_id}.mp4")
        npy_save_path = os.path.join(config.SAVE_NPY_DIR, f"{video_id}.npy")

        # 视频丢失检查
        if not os.path.exists(vid_path):
            missing_count += 1
            continue

        # =============================
        # 断点续传：如果已有 npy 就跳过提取，但要加入列表
        # =============================
        if not os.path.exists(npy_save_path):
            try:
                features = extract_features(vid_path, start_frame, end_frame)
                # 过滤空数据或极短数据
                if len(features) < 1: 
                    continue
                np.save(npy_save_path, features)
            except Exception as e:
                print(f"⚠️ 处理视频 {video_id} 失败: {e}")
                continue
        
        # 保存完整路径而不是 ID，这样 dataset.py 读取时不需要再拼路径，减少耦合
        subsets[subset].append(f"{npy_save_path},{label}")
        processed_count += 1

    # 输出统计
    print("\n📊 处理摘要:")
    print(f"   - 成功索引: {processed_count}")
    print(f"   - 缺失视频: {missing_count}")
    print(f"   - Train样本: {len(subsets['train'])}")
    print(f"   - Val样本:   {len(subsets['val'])}")
    print(f"   - Test样本:  {len(subsets['test'])}")

    # 保存 map 文件 (这些文件会被 Dataset 类直接读取)
    for subset_name, items in subsets.items():
        map_file = os.path.join(config.DATA_ROOT, f"{subset_name}_map_300.txt")
        with open(map_file, "w") as f:
            f.write("\n".join(items))
        print(f"💾 保存索引文件: {map_file}")

    # 计算统计量 (只用训练集)
    if len(subsets["train"]) > 0:
        mean, std = calculate_global_stats(subsets["train"])

        if mean is not None:
            np.save(os.path.join(config.DATA_ROOT, "global_mean_300.npy"), mean)
            np.save(os.path.join(config.DATA_ROOT, "global_std_300.npy"), std)
            print("💾 保存全局统计量 (global_mean_300.npy / global_std_300.npy)")
    else:
        print("⚠️ 警告：训练集为空，无法计算统计量！")

if __name__ == "__main__":
    process_dataset()