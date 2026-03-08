# 项目结构说明

本项目已重构为清晰的前后端分离结构。

## 目录结构

```
sign_language/
├── frontend/              # 前端项目 (React + Vite + TypeScript)
│   ├── src/              # 前端源代码
│   │   ├── pages/        # 页面组件
│   │   ├── shared/       # 共享组件
│   │   ├── styles/       # 样式文件
│   │   ├── types/        # TypeScript 类型定义
│   │   └── assets/       # 静态资源
│   ├── public/           # 公共资源
│   ├── package.json      # 前端依赖
│   ├── vite.config.ts    # Vite 配置
│   └── tsconfig.*.json   # TypeScript 配置
│
├── backend/              # 后端项目 (Python)
│   ├── src/             # 后端源代码
│   │   ├── config.py           # 配置文件
│   │   ├── model.py            # 模型定义
│   │   ├── dataset.py          # 数据集处理
│   │   ├── train.py            # 训练脚本
│   │   ├── inference.py        # 推理脚本
│   │   ├── preprocess.py       # 数据预处理
│   │   └── core_preprocess.py  # 核心预处理
│   └── requirements.txt  # Python 依赖
│
├── models/               # 机器学习模型代码
│   ├── S2HAND_code/     # 手部检测模型
│   └── 3d_sign_language/ # 3D 手语处理
│       ├── mano_v1_2/   # MANO 手部模型
│       └── src_3d/      # 3D 处理源码
│
├── notebooks/            # Jupyter Notebooks
│   └── train.ipynb      # 训练笔记本
│
├── outputs/              # 输出和临时文件
│   ├── debug_results/   # 调试结果
│   └── image/           # 图片输出
│
├── .gitignore           # Git 忽略配置
└── README.md            # 项目说明
```

## 启动指南

### 前端启动

```bash
cd frontend
npm install          # 首次运行需要安装依赖
npm run dev          # 启动开发服务器
```

访问: http://localhost:5173

### 后端启动

```bash
cd backend
pip install -r requirements.txt  # 首次运行需要安装依赖
python src/train.py              # 运行训练
python src/inference.py          # 运行推理
```

### 训练模型

```bash
cd notebooks
jupyter notebook train.ipynb    # 打开训练笔记本
```

## 注意事项

- 前端和后端现在完全分离，可以独立开发和部署
- 模型代码集中在 `models/` 目录，便于管理
- 数据集、模型权重等大文件请放在项目根目录的 `data/` 目录（已被 .gitignore 忽略）
- 输出结果统一放在 `outputs/` 目录
