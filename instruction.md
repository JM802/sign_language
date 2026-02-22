# 孤立词手语识别项目 (WLASL-300) 开发与设计文档

## 1. 项目概述
本项目是一个基于骨骼关键点（Skeleton-based）的孤立词手语识别系统，主要针对 WLASL (Word-Level American Sign Language) 数据集的 300 词子集（WLASL-300）进行开发。项目通过 MediaPipe 提取人体骨架关键点，构建了包含双重相对坐标与运动速度的复合特征体系，并使用带有注意力机制的双向 LSTM (BiLSTM + Attention) 模型进行序列分类。

## 2. 目录结构
```text
f:\sign_language\
├── data/                       # 数据集目录 (需手动放置 WLASL 数据)
│   ├── wlasl-complete/         # WLASL 原始数据
│   │   ├── videos/             # 原始 MP4 视频
│   │   └── nslt_300.json       # 官方 300 词划分文件
│   └── processed_features_300/ # MediaPipe 提取的 NPY 特征文件
├── result/                     # 模型保存输出目录
│   └── checkpoints/            # 训练阶段 Checkpoint 存放
├── src/                        # 源代码目录
│   ├── config.py               # 全局配置文件
│   ├── preprocess.py           # 视频预处理，特征提取与集划分
│   ├── core_preprocess.py      # 核心特征变换算法实现及全局统计量生成
│   ├── dataset.py              # PyTorch 数据集封装 (含数据增强与归一化)
│   ├── model.py                # 深度学习模型构建 (BiLSTM + Attention)
│   ├── train.py                # 训练、验证与测试的主循环代码
│   └── inference.py            # 推理脚本（待开发）
├── requirements.txt            # 项目依赖
└── instruction.md              # 本文档
```

## 3. 核心依赖
* `torch` / `torchvision`
* `numpy`
* `opencv-python` (cv2)
* `mediapipe`
* `tqdm`

## 4. 核心算法与特征工程
### 4.1 原始特征提取 (Input -> 134 维)
利用 MediaPipe Holistic 提取视频每帧的人体关键点（见 `preprocess.py`）：
* **上半身姿态 (Pose)**：取 0-24 号点（共25点），每个点含 x, y 坐标 $\rightarrow$ 50 维。
* **左手 (Left Hand)**：21 个关键点 $\rightarrow$  42 维。
* **右手 (Right Hand)**：21 个关键点 $\rightarrow$  42 维。
* **初始维度**：$50 + 42 + 42 = 134$ 维/帧。

### 4.2 核心特征变换 (134 维 -> 268 维)
单凭绝对坐标容易受到摄像头视角、距离与人物站位的影响。系统在 `core_preprocess.py` 和 `dataset.py` 中实现了 **“双重相对坐标 + 速度特征”** 的数据变换体系：
1. **双重相对坐标 (消除全局平移影响)**：
   * 上半身骨架 (Pose)：以**鼻子 (Nose/0号点)** 为基准重对齐。
   * 左手 (Left Hand)：以**左手腕 (Left Wrist/0号点)** 为基准重对齐。
   * 右手 (Right Hand)：以**右手腕 (Right Wrist/0号点)** 为基准重对齐。
2. **帧间运动速度 (引入动态时序信息)**：
   * 针对相对坐标计算相邻两帧的坐标差分 $(\Delta x, \Delta y)$，捕捉动作速度。
3. **最终特征组合**：
   * 相对坐标 (134 维) + 差分特征 (134 维) = **模型输入特征维数 268 维/帧**。

### 4.3 深度学习模型构建 (BiLSTM + Attention)
模型架构实现在 `model.py` 内部，接受形状为 `[Batch_Size, Seq_Len, 268]` 的输入特征流。
* **BiLSTM 层**：用于提取时间序列的长短时依赖，采用双向网络 (`bidirectional=True`)。
* **LayerNorm / Batch Normalization**：稳定并加速深层循环网络的训练。
* **Attention 注意力层**：网络针对各帧重要性并非均等。通过基于前馈层 (Tanh) + Softmax 的注意力机制，赋予关键过渡帧更高的权重，将时序信息聚合成高维 Context 向量。
* **分类头 (Classifier)**：通过全连接层激活与 Dropout 防止过拟合，并输出用于 `NUM_CLASSES=300` 类的预测分布。

## 5. 数据流向与生命周期 (I/O)

### 5.1 预处理与标注建立: `preprocess.py`
* **输入**：`data/wlasl-complete/videos/*.mp4`（原始MP4）和 `nslt_300.json` (官方数据集按Train/Val/Test标记的词汇)。
* **处理**：基于 MediaPipe 提取坐标，过滤残缺数据。
* **输出**：
  * 保存各视频的 `(T, 134)` NumPy 矩阵用于复用。
  * 生成 `train_map_300.txt`，`val_map_300.txt`，`test_map_300.txt`，建立“数据文件与手语Label”的映射索引。

### 5.2 全局特征分布记录: `core_preprocess.py`
* **处理**：遍历 `train_map` 生成的模型特征。
* **输出**：提前计算整个训练集 268 维的均值向量（`global_mean_300_double_vel.npy`）和标准差向量（`global_std_300_double_vel.npy`），以便在数据加载器中快速执行 Z-Score Normalization。

### 5.3 训练组装: `dataset.py` & `train.py`
* **动态 DataLoader (Dataset)**：
  * **Input**: 读取 `.npy` 与映射 txt，提取出原长度 T 帧的 134 维矩阵。
  * **Processing**: 实时执行上述的双重相对化和速度计算变为 268 维$\rightarrow$ 执行数据增强（Random Scale scaling 与 Random Gaussian Noise 添加）$\rightarrow$ 执行全局均值与方差化规整 $\rightarrow$ **时间帧统一对齐**（对过长视频下采样抽帧，对过短视频作 Zero-Padding 填充），保证序列长度锁定为 `SEQ_LEN=64`。
  * **Output**: `(64, 268)` 的 Tensor 给到模型。
* **模型优化器**: 采用 **Adam Optimizer + ReduceLROnPlateau** 策略，并计算 `CrossEntropyLoss` 进行梯度下降。

## 6. 后续开发指引 (Next Steps)
现有的算法和模型 pipeline 已经构建并跑通了一套强特征的训练测试结构，您可以接续做以下的开发计划：

1. **补全推理模块 (`inference.py`)**：
   * `inference.py` 目前为空。可以开发：加载最好 `best_model_300.pth` $\rightarrow$ OpenCV 调取电脑本地摄像头或读取某个陌生 mp4 $\rightarrow$ 调用 `preprocess.py` 的手段提点 $\rightarrow$ 执行 134 $\rightarrow$ 268 和 Padding 处理 $\rightarrow$ 进入 `model` 得到 Top-1 识别结果。
2. **算法性能迭代尝试**：
   * 将 `BiLSTM + Attention` 升级为 `Transformer Encoder`，处理变长输入以获取更强时序特征。
   * 采用 GCN (如 ST-GCN) 算法，专门针对 MediaPipe 的骨架连通图来挖掘空间拓扑结构关系。
3. **数据增强**：
   * 可以实现在时间维度的增强如 **动态扭曲(Time Warping)**、**随机抽帧抹去(Frame Dropping)** 帮助提升抗干扰性能。
