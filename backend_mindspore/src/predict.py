"""
predict.py —— 使用 MindSpore .ckpt 权重进行单样本预测

示例：
    python src/predict.py
    python src/predict.py --input backend/data/processed_features_300/toy_c0_test_00.npy
    python src/predict.py --checkpoint backend/result/checkpoints/best_model_300.ckpt --topk 5
"""

import argparse
import os
from typing import Tuple

import numpy as np
import mindspore as ms
from mindspore import context, load_checkpoint, load_param_into_net

import config
from core_preprocess import to_double_relative_with_velocity
from model import BiLSTMAttentionModel


def setup_device() -> str:
    """配置运行设备，Ascend 不可用时自动回退到 CPU。"""
    try:
        context.set_context(mode=context.GRAPH_MODE)
        if hasattr(ms, "set_device"):
            ms.set_device(config.DEVICE)
        else:
            context.set_context(device_target=config.DEVICE)
        if config.DEVICE == "Ascend":
            context.set_context(ascend_config={"precision_mode": "allow_fp32_to_fp16"})
        print(f"✅ Predict device: {config.DEVICE}")
        return config.DEVICE
    except Exception:
        context.set_context(mode=context.GRAPH_MODE)
        if hasattr(ms, "set_device"):
            ms.set_device("CPU")
        else:
            context.set_context(device_target="CPU")
        print("⚠️  Ascend not available, fallback to CPU")
        return "CPU"


def load_normalization() -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(config.GLOBAL_MEAN_PATH) or not os.path.exists(config.GLOBAL_STD_PATH):
        raise FileNotFoundError(
            "归一化参数不存在，请先准备数据集或运行 generate_toy_dataset.py。"
        )

    mean = np.load(config.GLOBAL_MEAN_PATH).astype(np.float32)
    std = np.load(config.GLOBAL_STD_PATH).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def resolve_input_path(npy_path: str) -> str:
    """兼容旧 map 中的绝对路径，自动回退到当前数据目录。"""
    if os.path.exists(npy_path):
        return npy_path

    fname = os.path.basename(npy_path)
    candidates = [
        os.path.join(config.PROCESSED_FEATURE_DIR, fname),
        os.path.join(config.DATA_ROOT, fname),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(f"找不到输入文件: {npy_path}")


def preprocess_npy(npy_path: str, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    读取单个 .npy 样本并处理为模型输入。

    支持两种输入：
    - (T, 134): 原始坐标特征
    - (T, 268): 已处理好的双相对坐标+速度特征
    """
    npy_path = resolve_input_path(npy_path)

    data = np.load(npy_path).astype(np.float32)

    if data.ndim != 2:
        raise ValueError(f"输入数据维度错误，期望二维数组，实际为: {data.shape}")

    if data.shape[1] == 134:
        data = to_double_relative_with_velocity(data)
    elif data.shape[1] != 268:
        raise ValueError(
            f"输入特征维度不支持，期望 134 或 268，实际为: {data.shape[1]}"
        )

    data = (data - mean) / std

    seq_len = config.SEQ_LEN
    if data.shape[0] > seq_len:
        idxs = np.linspace(0, data.shape[0] - 1, seq_len, dtype=int)
        data = data[idxs]
    elif data.shape[0] < seq_len:
        pad = np.zeros((seq_len - data.shape[0], data.shape[1]), dtype=np.float32)
        data = np.concatenate([data, pad], axis=0)

    return data.astype(np.float32)


def pick_default_sample() -> str:
    """默认取 test_map 中第一条样本，方便直接演示预测。"""
    if not os.path.exists(config.TEST_MAP_PATH):
        raise FileNotFoundError(
            f"找不到测试映射文件: {config.TEST_MAP_PATH}，请先生成数据。"
        )

    with open(config.TEST_MAP_PATH, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    if not first_line:
        raise ValueError(f"测试映射文件为空: {config.TEST_MAP_PATH}")

    sample_path = first_line.split(",")[0].strip()
    return sample_path


def build_model(checkpoint_path: str, runtime_device: str) -> BiLSTMAttentionModel:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到权重文件: {checkpoint_path}")

    model = BiLSTMAttentionModel(
        input_size=config.INPUT_SIZE,
        hidden_size=256,
        num_classes=config.NUM_CLASSES,
    )
    if runtime_device == "Ascend":
        model.to_float(ms.float16)
    param_dict = load_checkpoint(checkpoint_path)
    load_param_into_net(model, param_dict)
    model.set_train(False)
    return model


def predict(model: BiLSTMAttentionModel, sample: np.ndarray, topk: int = 3, runtime_device: str = "CPU"):
    input_dtype = ms.float16 if runtime_device == "Ascend" else ms.float32
    np_dtype = np.float16 if runtime_device == "Ascend" else np.float32
    tensor = ms.Tensor(sample[None, ...].astype(np_dtype), dtype=input_dtype)  # [1, T, 268]
    logits = model(tensor)
    probs = ms.ops.softmax(logits, axis=1).asnumpy()[0]

    topk = max(1, min(topk, probs.shape[0]))
    top_indices = np.argsort(-probs)[:topk]
    return probs, top_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 .ckpt 权重进行单样本预测")
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="输入 .npy 文件路径；不传则默认取 test_map_300.txt 第一条样本",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=config.BEST_MODEL_PATH,
        help="MindSpore checkpoint 路径",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="输出前 K 个预测类别",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_device = setup_device()

    sample_path = args.input or pick_default_sample()
    mean, std = load_normalization()
    sample = preprocess_npy(sample_path, mean, std)
    model = build_model(args.checkpoint, runtime_device)
    probs, top_indices = predict(model, sample, args.topk, runtime_device)

    print(f"\n📄 输入样本: {sample_path}")
    print(f"📦 权重文件: {args.checkpoint}")
    print(f"📐 模型输入形状: {sample.shape}")
    print("\n🔮 Top-K 预测结果:")
    for rank, cls_id in enumerate(top_indices, start=1):
        print(f"  {rank}. class={int(cls_id):3d}, prob={probs[cls_id]:.6f}")

    print(f"\n✅ 最终预测类别: {int(top_indices[0])}")


if __name__ == "__main__":
    main()
