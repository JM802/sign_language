"""
train.py —— MindSpore 版训练脚本 (适配昇腾 Ascend)
"""
import os
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore import context, save_checkpoint, load_checkpoint, load_param_into_net
import config
from dataset import WLASLDataset
from model import BiLSTMAttentionModel

# ===================== 设备 & 全局配置 =====================
# 昇腾环境使用 Ascend；无卡时回退到 CPU
RUNTIME_DEVICE = config.DEVICE
try:
    context.set_context(mode=context.GRAPH_MODE)
    if hasattr(ms, "set_device"):
        ms.set_device(config.DEVICE)
    else:
        context.set_context(device_target=config.DEVICE)
    if config.DEVICE == "Ascend":
        context.set_context(ascend_config={"precision_mode": "allow_fp32_to_fp16"})
    print(f"✅ MindSpore device: {config.DEVICE}")
except Exception:
    RUNTIME_DEVICE = "CPU"
    context.set_context(mode=context.GRAPH_MODE)
    if hasattr(ms, "set_device"):
        ms.set_device("CPU")
    else:
        context.set_context(device_target="CPU")
    print("⚠️  Ascend not available, fallback to CPU")

NUM_WORKERS = 1 if os.name == "nt" else 4
argmax = ops.Argmax(axis=1)

BATCH_SIZE  = config.BATCH_SIZE
EPOCHS      = config.EPOCHS
SEQ_LEN     = config.SEQ_LEN
NUM_CLASSES = config.NUM_CLASSES
INPUT_SIZE  = 268  # 双重相对坐标 + 速度
INPUT_DTYPE = ms.float16 if RUNTIME_DEVICE == "Ascend" else ms.float32
LABEL_DTYPE = ms.int32

# ===================== 数据集 =====================
train_set = WLASLDataset(config.TRAIN_MAP_PATH, mode='train')
val_set   = WLASLDataset(config.VAL_MAP_PATH,   mode='val')
test_set  = WLASLDataset(config.TEST_MAP_PATH,  mode='test')

# GeneratorDataset 需指定列名
train_loader = ds.GeneratorDataset(
    source=train_set,
    column_names=["data", "label"],
    shuffle=True,
    num_parallel_workers=NUM_WORKERS
).batch(BATCH_SIZE, drop_remainder=False)

val_loader = ds.GeneratorDataset(
    source=val_set,
    column_names=["data", "label"],
    shuffle=False,
    num_parallel_workers=NUM_WORKERS
).batch(BATCH_SIZE, drop_remainder=False)

test_loader = ds.GeneratorDataset(
    source=test_set,
    column_names=["data", "label"],
    shuffle=False,
    num_parallel_workers=NUM_WORKERS
).batch(BATCH_SIZE, drop_remainder=False)

# ===================== 模型 =====================
model = BiLSTMAttentionModel(
    input_size=INPUT_SIZE,
    hidden_size=256,
    num_classes=NUM_CLASSES
)
if RUNTIME_DEVICE == "Ascend":
    model.to_float(ms.float16)

# ===================== 损失 & 优化器 =====================
loss_fn   = nn.CrossEntropyLoss()
optimizer = nn.Adam(model.trainable_params(), learning_rate=config.LEARNING_RATE)


# ===================== 前向 + 损失 封装 =====================
class WithLossCell(nn.Cell):
    """将 model + loss_fn 合并为一个 Cell，方便 TrainOneStepCell 使用"""
    def __init__(self, backbone, loss_fn):
        super().__init__()
        self.backbone = backbone
        self.loss_fn = loss_fn

    def construct(self, data, label):
        logits = self.backbone(data)
        return self.loss_fn(logits, label)


net_with_loss = WithLossCell(model, loss_fn)
train_cell = nn.TrainOneStepCell(net_with_loss, optimizer)
train_cell.set_train(True)


# ===================== 辅助函数 =====================
def evaluate(network, data_loader):
    """在 data_loader 上评估准确率"""
    network.set_train(False)
    correct, total = 0, 0
    for batch in data_loader.create_dict_iterator():
        data  = ops.cast(batch["data"], INPUT_DTYPE)
        label = ops.cast(batch["label"], LABEL_DTYPE)
        logits = network(data)
        preds = argmax(logits)
        correct += (preds == label).asnumpy().sum()
        total   += label.shape[0]
    return correct / total if total > 0 else 0.0


# ===================== 训练循环 =====================
best_val_acc = -1.0
best_model_path = config.BEST_MODEL_PATH
last_model_path = config.LAST_MODEL_PATH

# 学习率衰减：验证精度 patience 轮不提升则减半
lr_patience   = 5
lr_wait       = 0
current_lr    = config.LEARNING_RATE

for epoch in range(EPOCHS):
    # —— 训练阶段 ——
    train_cell.set_train(True)
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for batch in train_loader.create_dict_iterator():
        data  = ops.cast(batch["data"], INPUT_DTYPE)
        label = ops.cast(batch["label"], LABEL_DTYPE)
        loss  = train_cell(data, label)

        # 统计
        total_loss += loss.asnumpy().item() * label.shape[0]
        logits = model(data)
        preds  = argmax(logits)
        total_correct += (preds == label).asnumpy().sum()
        total_samples += label.shape[0]

    train_loss = total_loss / total_samples
    train_acc  = total_correct / total_samples

    # —— 验证阶段 ——
    val_acc = evaluate(model, val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS}: "
          f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
          f"Val Acc={val_acc:.4f}, LR={current_lr:.6f}")

    # —— 学习率调度 (ReduceOnPlateau 手动实现) ——
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        lr_wait = 0
        # 保存最优 checkpoint
        save_checkpoint(model, best_model_path)
        print(f"✅ Saved best model at epoch {epoch+1}, Val Acc={val_acc:.4f}")
    else:
        lr_wait += 1
        if lr_wait >= lr_patience:
            current_lr *= 0.5
            lr_wait = 0
            # 重建优化器以应用新学习率
            optimizer = nn.Adam(model.trainable_params(), learning_rate=current_lr)
            train_cell = nn.TrainOneStepCell(net_with_loss, optimizer)
            train_cell.set_train(True)
            print(f"📉 LR reduced to {current_lr:.6f}")

    save_checkpoint(model, last_model_path)

# ===================== 测试 =====================
print("===== Test on best_model =====")
param_dict = load_checkpoint(best_model_path)
load_param_into_net(model, param_dict)
test_acc = evaluate(model, test_loader)
print(f"🎯 Test Accuracy: {test_acc:.4f}")
