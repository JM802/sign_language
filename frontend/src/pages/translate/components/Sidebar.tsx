type TranslateMode = "camera" | "video" | "help";

interface SidebarProps {
  currentMode: TranslateMode;
  onModeChange: (mode: TranslateMode) => void;
  isRecording: boolean;
  isCameraActive: boolean;
  isProcessing: boolean;
}

/**
 * 左侧边栏 - 对应 PyQt 中的 frame_2 侧边栏
 * 包含 Logo、模式切换、Start/Pause 按钮
 */
export default function Sidebar({
  currentMode,
  onModeChange,
  isRecording,
  isCameraActive,
  isProcessing,
}: SidebarProps) {
  const modes: { key: TranslateMode; icon: string; label: string }[] = [
    { key: "video", icon: "🎬", label: "Video" },
    { key: "camera", icon: "📷", label: "Camera" },
    { key: "help", icon: "❓", label: "Help" },
  ];

  return (
    <div className="translate-sidebar">
      <div className="sidebar-brand">
        <span className="sidebar-brand-kicker">gesture ai</span>
        <h2 className="sidebar-brand-title">Translate</h2>
        <p className="sidebar-brand-subtitle">
          实时识别手语动作，并输出文本与语音结果。
        </p>
      </div>

      <nav>
        {modes.map((m) => (
          <button
            key={m.key}
            type="button"
            className={`sidebar-nav-item ${currentMode === m.key ? "active" : ""}`}
            onClick={() => onModeChange(m.key)}
          >
            <span className="sidebar-nav-icon">{m.icon}</span>
            <span className="sidebar-nav-label">{m.label}</span>
          </button>
        ))}
      </nav>

      <div className="sidebar-status-row">
        <span
          className={`sidebar-status-dot ${isRecording ? "recording" : isCameraActive ? "online" : "idle"}`}
        />
        <span className="sidebar-status-text">
          {isProcessing
            ? "模型处理中..."
            : isRecording
              ? "正在录制手语动作"
              : isCameraActive
                ? "摄像头已就绪"
                : "等待选择模式"}
        </span>
      </div>
    </div>
  );
}

export type { TranslateMode };
