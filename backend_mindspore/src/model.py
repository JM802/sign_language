import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

class Attention(nn.Cell):
    """注意力机制 — 对 GRU 输出做加权聚合"""
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # 单向GRU：输入维度为 hidden_size
        self.dense1 = nn.Dense(hidden_size, hidden_size // 2)
        self.tanh = nn.Tanh()
        self.dense2 = nn.Dense(hidden_size // 2, 1)

    def construct(self, gru_output):
        # gru_output: [B, Seq, Hidden] (单向)
        scores = self.dense2(self.tanh(self.dense1(gru_output)))  # [B, Seq, 1]
        weights = ops.softmax(scores, axis=1)
        context = ops.reduce_sum(weights * gru_output, axis=1)    # [B, Hidden]
        return context


class BiLSTMAttentionModel(nn.Cell):
    """单向GRU + Attention 手语识别模型 (Ascend优化版)"""
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2, dropout=0.3):
        super(BiLSTMAttentionModel, self).__init__()
        # 改为GRU，解决Ascend NPU的DynamicRNN算子编译失败问题
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,  # 单向GRU
            dropout=float(dropout) if num_layers > 1 else 0.0
        )
        self.ln = nn.LayerNorm((hidden_size,))  # 单向：hidden_size
        self.attention = Attention(hidden_size)
        self.fc1 = nn.Dense(hidden_size, hidden_size // 2)
        self.bn1 = nn.BatchNorm1d(hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Dense(hidden_size // 2, num_classes)

    def construct(self, x):
        # x: [B, Seq, InputSize]
        out, _ = self.gru(x)   # [B, Seq, Hidden] (单向GRU)
        out = self.ln(out)
        context = self.attention(out)
        out = self.fc1(context)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


if __name__ == "__main__":
    # 简单测试模型前向
    import os

    # 检测设备并配置
    try:
        # 尝试使用Ascend
        ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")
        ms.set_context(ascend_config={"precision_mode": "allow_fp32_to_fp16"})
        device = "Ascend"
        dtype = ms.float16
        print("🔧 Testing on Ascend with FP16")
    except:
        # 回退到CPU
        ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
        device = "CPU"
        dtype = ms.float32
        print("🔧 Testing on CPU with FP32")

    batch_size = 2
    seq_len = 64
    input_size = 268
    num_classes = 300

    model = BiLSTMAttentionModel(input_size, hidden_size=256, num_classes=num_classes)

    # Ascend设备转换为FP16
    if device == "Ascend":
        model.to_float(ms.float16)

    dummy_input = ms.Tensor(np.random.rand(batch_size, seq_len, input_size).astype(np.float16 if dtype == ms.float16 else np.float32), dtype=dtype)
    output = model(dummy_input)

    print(f"✅ Output shape: {output.shape}")  # 应为 [2, 300]
    print(f"✅ 单向GRU模型测试成功！({device} - {dtype})")
