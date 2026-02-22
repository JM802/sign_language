import numpy as np
import os
from tqdm import tqdm
import config

# =======================================================
# 方案三 + 方案四：双重相对坐标 + 速度特征
# 输入 134 -> 输出 268 (含速度)
# =======================================================
def to_double_relative_with_velocity(data):
    """
    data: (T, 134)
    return: (T, 268)  # [pose_rel, lh_rel, rh_rel, d_pose, d_lh, d_rh]
    """

    T = data.shape[0]

    # ---------------------------
    # 拆解为 (x,y) 坐标
    # ---------------------------
    pose = data[:, 0:50].reshape(T, 25, 2)
    lh   = data[:, 50:92].reshape(T, 21, 2)
    rh   = data[:, 92:134].reshape(T, 21, 2)

    # ---------------------------
    # 方案三：双重相对坐标
    # ---------------------------
    nose = pose[:, 0:1, :]      # 基准：鼻子
    l_wrist = lh[:, 0:1, :]     # 基准：左手腕
    r_wrist = rh[:, 0:1, :]     # 基准：右手腕

    pose_rel = pose - nose
    lh_rel = lh - l_wrist
    rh_rel = rh - r_wrist

    # ---------------------------
    # 方案四：帧间速度 Dx Dy
    # ---------------------------
    pose_d = np.diff(pose_rel, axis=0)
    lh_d   = np.diff(lh_rel,   axis=0)
    rh_d   = np.diff(rh_rel,   axis=0)

    # 补第一帧的速度 = 0
    pose_d = np.concatenate([np.zeros_like(pose_d[:1]), pose_d], axis=0)
    lh_d   = np.concatenate([np.zeros_like(lh_d[:1]),   lh_d],   axis=0)
    rh_d   = np.concatenate([np.zeros_like(rh_d[:1]),   rh_d],   axis=0)

    # ---------------------------
    # 展平：每帧 268 维
    # ---------------------------
    final_feat = np.concatenate([
        pose_rel.reshape(T, -1),    # 50
        lh_rel.reshape(T, -1),      # 42
        rh_rel.reshape(T, -1),      # 42
        pose_d.reshape(T, -1),      # 50
        lh_d.reshape(T, -1),        # 42
        rh_d.reshape(T, -1)         # 42
    ], axis=1)

    return final_feat  # (T, 268)



# =======================================================
# 主流程：遍历 train_map_300.txt → 转换 → 计算 mean/std
# =======================================================
def main():
    print("🔥 开始生成：双相对坐标 + 速度特征 的全局统计量...")
    
    train_map = os.path.join(config.DATA_ROOT, "train_map_300.txt")
    if not os.path.exists(train_map):
        print(f"❌ 找不到 {train_map}")
        return

    with open(train_map, 'r') as f:
        lines = f.readlines()

    all_data = []

    for line in tqdm(lines):
        path_str = line.split(',')[0].strip()
        fname = os.path.basename(path_str)

        # 先从 processed_features_300 找
        npy_path = os.path.join(config.DATA_ROOT, "processed_features_300", fname)
        if not os.path.exists(npy_path):
            # 再从 root 找
            npy_path = os.path.join(config.DATA_ROOT, fname)
            if not os.path.exists(npy_path):
                continue

        try:
            raw = np.load(npy_path).astype(np.float32)

            # 🔥 新增：双重相对坐标 + 速度
            rel_vel = to_double_relative_with_velocity(raw)
            all_data.append(rel_vel)
        except:
            continue

    if not all_data:
        print("❌ 没有数据可计算 mean/std")
        return

    print("📌 拼接全部数据...")
    big_data = np.concatenate(all_data, axis=0)  # (N, 268)

    print("📌 计算 mean/std...")
    mean = np.mean(big_data, axis=0)
    std  = np.std(big_data, axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    # 保存
    save_mean = os.path.join(config.DATA_ROOT, "global_mean_300_double_vel.npy")
    save_std  = os.path.join(config.DATA_ROOT, "global_std_300_double_vel.npy")

    np.save(save_mean, mean)
    np.save(save_std,  std)

    print("\n🎉 完成！已生成：")
    print(f"   {save_mean}")
    print(f"   {save_std}")
    print("   (维度：268) ✔")


if __name__ == "__main__":
    main()
