# Frontend - 手语识别前端

基于 React + Vite + TypeScript 的现代化前端应用。

## 技术栈

- **React 19** - UI 框架
- **TypeScript** - 类型安全
- **Vite** - 构建工具
- **React Router** - 路由管理
- **Three.js** - 3D 渲染
- **GSAP** - 动画库

## 开发环境设置

### 安装依赖

```bash
npm install
```

### 启动开发服务器

```bash
npm run dev
```

访问 http://localhost:5173

### 构建生产版本

```bash
npm run build
```

生成的文件位于 `dist/` 目录

### 预览生产构建

```bash
npm run preview
```

### 代码检查

```bash
npm run lint
```

## 项目结构

```
src/
├── pages/              # 页面组件
│   ├── home/          # 首页
│   ├── learning/      # 学习页面
│   └── about/         # 关于页面
├── shared/            # 共享组件
│   └── components/    # 通用组件
├── styles/            # 全局样式
├── types/             # TypeScript 类型定义
├── assets/            # 静态资源
├── App.tsx            # 根组件
└── main.tsx           # 入口文件
```

## 开发规范

- 使用 TypeScript 进行类型检查
- 遵循 ESLint 配置的代码规范
- 组件使用函数式组件 + Hooks
- 样式文件与组件放在同一目录

## 环境要求

- Node.js >= 18
- npm >= 9
