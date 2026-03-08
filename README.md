# Sign Language Translator

智能手语识别与翻译系统，支持实时手语识别、3D 手部建模和交互式学习。

## 🌟 功能特点

- **实时手语识别** - 基于深度学习的手语动作识别
- **3D 手部建模** - 使用 MANO 模型进行精确的手部姿态估计
- **交互式学习** - 友好的 Web 界面，支持手语学习和练习
- **高性能** - 前后端分离架构，支持高并发访问

## 📁 项目结构

```
sign_language/
├── frontend/          # React + Vite + TypeScript 前端
├── backend/           # Python 机器学习后端
├── models/            # ML 模型代码 (S2HAND, MANO)
├── notebooks/         # Jupyter 训练笔记本
└── outputs/           # 输出和调试文件
```

详细结构说明请查看 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

## 🚀 快速开始

### 前端启动

```bash
cd frontend
npm install
npm run dev
```

访问: http://localhost:5173

### 后端启动

```bash
cd backend
pip install -r requirements.txt
python src/inference.py
```

## 📚 文档

- [项目结构说明](PROJECT_STRUCTURE.md)
- [前端开发文档](frontend/README.md)
- [后端开发文档](backend/README.md)

## 🛠️ 技术栈

### 前端
- React 19
- TypeScript
- Vite
- Three.js (3D 渲染)
- GSAP (动画)

### 后端
- Python
- PyTorch
- OpenCV
- NumPy

### 模型
- S2HAND (手部检测)
- MANO (手部建模)
- 自定义手语识别模型

## 📝 开发计划

查看各模块的开发笔记：
- Backend: `backend/src/next_step.md`
- 3D Processing: `models/3d_sign_language/src_3d/thinking_3d.md`

## 📄 许可证

本项目用于学习和研究目的。
