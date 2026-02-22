# src/dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
import config
from core_preprocess import to_double_relative, add_velocity

class WLASLDataset(Dataset):
    def __init__(self, map_file, mode='train'):
        """
        map_file: train_map_300.txt / val_map_300.txt / test_map_300.txt
        mode: 'train' / 'val' / 'test'
        """
        self.mode = mode
        self.seq_len = config.SEQ_LEN  # 64

        # 读取映射文件
        with open(map_file, 'r') as f:
            self.lines = f.readlines()

        # 归一化参数初始化
        self.mean = None
        self.std  = None

        # 自动查找 double_vel mean/std
        mean_name = "global_mean_300_double_vel.npy"
        std_name  = "global_std_300_double_vel.npy"
        mean_path = os.path.join(config.DATA_ROOT, mean_name)
        std_path  = os.path.join(config.DATA_ROOT, std_name)
        if os.path.exists(mean_path):
            self.mean = np.load(mean_path).astype(np.float32)
            self.std  = np.load(std_path).astype(np.float32)
            print(f"[{self.mode}] Loaded normalization: {mean_path}")
        else:
            print(f"[{self.mode}] Warning: *_double_vel normalization not found!")

    def set_normalization(self, mean, std):
        """外部注入归一化参数"""
        self.mean = mean.astype(np.float32)
        self.std  = std.astype(np.float32)
        print(f"[{self.mode}] Normalization injected via set_normalization()")

    def __len__(self):
        return len(self.lines)

    def _augment(self, data134):
        # 缩放 + 高斯噪声
        scale = random.uniform(0.90, 1.10)
        data134 = data134 * scale
        noise = np.random.normal(0, 0.005, data134.shape).astype(np.float32)
        data134 = data134 + noise
        return data134.astype(np.float32)

    def __getitem__(self, idx):
        line = self.lines[idx].strip()
        npy_path, label = line.split(',')
        fname = os.path.basename(npy_path)
        full_path = os.path.join(config.DATA_ROOT, "processed_features_300", fname)
        if not os.path.exists(full_path):
            full_path = os.path.join(config.DATA_ROOT, fname)

        try:
            raw = np.load(full_path).astype(np.float32)
        except:
            # 文件损坏处理
            return torch.zeros((self.seq_len, 268)), torch.tensor(0)

        # 双重相对坐标
        data = to_double_relative(raw)

        # 数据增强 (训练集才做)
        if self.mode == 'train':
            data = self._augment(data)

        # 计算速度
        data = add_velocity(data)  # (T, 268)

        # 归一化
        if self.mean is not None:
            data = (data - self.mean) / self.std

        # 序列长度统一
        T = data.shape[0]
        if T > self.seq_len:
            idxs = np.linspace(0, T - 1, self.seq_len, dtype=int)
            data = data[idxs]
        elif T < self.seq_len:
            pad = np.zeros((self.seq_len - T, data.shape[1]), dtype=np.float32)
            data = np.concatenate([data, pad], axis=0)

        return torch.from_numpy(data), torch.tensor(int(label), dtype=torch.long)
