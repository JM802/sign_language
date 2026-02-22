# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
import config
from dataset import WLASLDataset
from model import BiLSTMAttentionModel

# ========================== 全局配置 ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = config.BATCH_SIZE
EPOCHS = config.EPOCHS
SEQ_LEN  = config.SEQ_LEN
NUM_CLASSES = config.NUM_CLASSES
INPUT_SIZE = 268  # 双重相对 + 速度

# ========================== 数据集 ==========================
train_set = WLASLDataset(os.path.join(config.DATA_ROOT, "train_map_300.txt"), mode='train')
val_set   = WLASLDataset(os.path.join(config.DATA_ROOT, "val_map_300.txt"), mode='val')
test_set  = WLASLDataset(os.path.join(config.DATA_ROOT, "test_map_300.txt"), mode='test')

# 可选：手动注入归一化
# mean, std = compute_global_normalization(train_set)
# train_set.set_normalization(mean, std)
# val_set.set_normalization(mean, std)
# test_set.set_normalization(mean, std)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

# ========================== 模型 ==========================
model = BiLSTMAttentionModel(input_size=INPUT_SIZE, hidden_size=256, num_classes=NUM_CLASSES).to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5, verbose=True)

# ========================== 训练 ==========================
best_val_acc = 0
best_model_path = os.path.join(config.RESULT_DIR, "best_model_300.pth")
for epoch in range(EPOCHS):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        total_correct += (out.argmax(dim=1) == y).sum().item()
        total_samples += y.size(0)
    train_loss = total_loss / total_samples
    train_acc  = total_correct / total_samples

    # ========================== 验证 ==========================
    model.eval()
    val_correct, val_samples = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            val_correct += (out.argmax(dim=1) == y).sum().item()
            val_samples += y.size(0)
    val_acc = val_correct / val_samples

    scheduler.step(val_acc)
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

    # 保存 best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ Saved best model at epoch {epoch+1}, Val Acc={val_acc:.4f}")

# ========================== 测试 ==========================
print("===== Test on best_model =====")
model.load_state_dict(torch.load(best_model_path))
model.eval()
test_correct, test_samples = 0, 0
with torch.no_grad():
    for x, y in tqdm(test_loader, desc="Testing"):
        x, y = x.to(device), y.to(device)
        out = model(x)
        test_correct += (out.argmax(dim=1) == y).sum().item()
        test_samples += y.size(0)
test_acc = test_correct / test_samples
print(f"🎯 Test Accuracy: {test_acc:.4f}")
