# Backend - 手语识别后端

基于 Python 的机器学习后端，负责模型训练、数据预处理和推理。

## 项目结构

```
src/
├── config.py           # 配置文件
├── model.py            # 模型定义
├── dataset.py          # 数据集处理
├── train.py            # 训练脚本
├── inference.py        # 推理脚本
├── preprocess.py       # 数据预处理
├── core_preprocess.py  # 核心预处理逻辑
├── instruction.md      # 使用说明
├── next_step.md        # 开发计划
└── thinkng.md          # 开发笔记
```

## 环境设置

### 安装依赖

```bash
pip install -r requirements.txt
```

建议使用虚拟环境：

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 数据预处理

```bash
python src/preprocess.py
```

### 训练模型

```bash
python src/train.py
```

### 运行推理

```bash
python src/inference.py
```

## 配置

编辑 `src/config.py` 文件来修改：
- 数据集路径
- 模型参数
- 训练超参数
- 输出路径

## 数据集

数据集应放在项目根目录的 `data/` 目录下（已被 .gitignore 忽略）。

## 模型输出

训练好的模型权重会保存在 `checkpoints/` 目录（已被 .gitignore 忽略）。

## 开发笔记

- 详细的使用说明见 `src/instruction.md`
- 下一步开发计划见 `src/next_step.md`
- 开发过程记录见 `src/thinkng.md`
