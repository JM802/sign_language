import { useState, useCallback, useMemo } from "react";
import Sidebar from "./components/Sidebar";
import type { TranslateMode } from "./components/Sidebar";
import ScannerAnimation from "./components/ScannerAnimation";
import CameraView from "./components/CameraView";
import VideoUpload from "./components/VideoUpload";
import HelpPanel from "./components/HelpPanel";
import ResultPanel from "./components/ResultPanel";
import { useCamera } from "./hooks/useCamera";
import { useVoice } from "./hooks/useVoice";
import { useTranslation } from "./hooks/useTranslation";
import "../../styles/translate.css";

/**
 * 翻译页面 - 对应 PyQt 中的 MyMainForm 主窗口
 * 三栏布局: 侧边栏 | 主视图区 | 结果面板
 */
export default function Translate() {
  const [mode, setMode] = useState<TranslateMode>("camera");
  const camera = useCamera();
  const voice = useVoice();
  const translation = useTranslation();
  const latestResult =
    translation.results[translation.results.length - 1] ?? null;
  const {
    isActive,
    isStarting,
    isRecording,
    frameCount,
    error,
    startCamera,
    stopCamera,
    toggleRecording,
  } = camera;
  const cleanedLatestText = useMemo(() => {
    if (!latestResult) {
      return "";
    }

    return latestResult.text
      .replace(/^识别结果[:：]?\s*/u, "")
      .replace(/\(置信度[:：]?[^)]*\)/gu, "")
      .split("\n")[0]
      .trim();
  }, [latestResult]);

  const displayTitle = useMemo(() => {
    if (translation.isProcessing) {
      return "Analyzing...";
    }

    if (isRecording) {
      return "Capturing...";
    }

    if (mode === "camera" && error && !isActive) {
      return "Camera Error";
    }

    if (cleanedLatestText) {
      return cleanedLatestText;
    }

    return mode === "camera"
      ? "Ready"
      : mode === "video"
        ? "Upload Video"
        : "Guide";
  }, [
    cleanedLatestText,
    error,
    isActive,
    isRecording,
    mode,
    translation.isProcessing,
  ]);

  const titleVariant = useMemo(() => {
    const length = displayTitle.length;

    if (length > 28) return "translate-title title-xlong";
    if (length > 16) return "translate-title title-long";
    return "translate-title";
  }, [displayTitle]);

  const heroSubtitle = useMemo(() => {
    if (translation.isProcessing) {
      return "模型正在分析刚刚采集到的动作序列，识别结果会在完成后立即刷新。";
    }

    if (isRecording) {
      return "正在录制手语动作，请保持动作完整，结束后会自动进入识别流程。";
    }

    if (mode === "camera" && error) {
      return error;
    }

    if (latestResult) {
      return `最新识别结果已同步显示。当前置信度 ${(latestResult.confidence * 100).toFixed(1)}%。`;
    }

    return "把摄像头里的手语动作，实时转换成清晰的文本反馈与语音播报。";
  }, [error, isRecording, latestResult, mode, translation.isProcessing]);

  const summaryText = useMemo(() => {
    if (translation.isProcessing) {
      return "正在执行时序分析与动作分类，请稍候，结果会自动写入右侧记录面板。";
    }

    if (isRecording) {
      return "已进入录制状态，系统正在持续采集摄像头画面。完成手语动作后点击停止即可识别。";
    }

    if (latestResult) {
      return latestResult.text;
    }

    if (mode === "video") {
      return "上传一个视频文件，系统会基于后端推理接口返回识别文本与置信度。";
    }

    if (mode === "camera") {
      return (
        error ?? "暂无识别结果，点击右侧按钮开始录制，或切换视频模式上传样例。"
      );
    }

    return "查看帮助说明后，可以切换到 Camera 或 Video 模式开始体验翻译功能。";
  }, [error, isRecording, latestResult, mode, translation.isProcessing]);

  const liveStatusLabel = useMemo(() => {
    if (translation.isProcessing) return "分析中";
    if (isRecording) return "录制中";
    if (latestResult) return "结果已更新";
    if (mode === "video") return "等待上传";
    if (mode === "camera" && isActive) return "实时待命";
    return "空闲";
  }, [isActive, isRecording, latestResult, mode, translation.isProcessing]);

  // 模式切换处理 - 对应 PyQt on_sidebar_click
  const handleModeChange = useCallback(
    async (newMode: TranslateMode) => {
      // 切离摄像头模式时关闭摄像头
      if (mode === "camera" && newMode !== "camera") {
        stopCamera();
      }
      setMode(newMode);
    },
    [mode, stopCamera],
  );

  // 摄像头 video ref 回调
  const handleVideoRef = useCallback(
    async (el: HTMLVideoElement) => {
      if (!isActive && !isStarting) {
        try {
          await startCamera(el);
        } catch {
          // 保持当前模式，在视图区给出错误和重试入口
        }
      }
    },
    [isActive, isStarting, startCamera],
  );

  // 录制切换处理 - 对应 PyQt camera_thread.toggle_recording
  const handleToggleRecording = useCallback(async () => {
    const blob = await toggleRecording();
    if (blob) {
      // 停止录制返回了数据，开始推理
      const result = await translation.inferFromBlob(blob);
      if (result) {
        voice.speak(result.text);
      }
    }
  }, [toggleRecording, translation, voice]);

  // 视频文件处理 - 对应 PyQt handle_video_import
  const handleFileSelected = useCallback(
    async (file: File) => {
      const result = await translation.inferFromFile(file);
      if (result) {
        voice.speak(result.text);
      }
    },
    [translation, voice],
  );

  const handlePrimaryAction = useCallback(async () => {
    if (translation.isProcessing) {
      return;
    }

    if (mode === "help") {
      setMode("camera");
      return;
    }

    if (mode === "camera") {
      await handleToggleRecording();
    }
  }, [handleToggleRecording, mode, translation.isProcessing]);

  const primaryActionLabel = translation.isProcessing
    ? "Processing..."
    : mode === "help"
      ? "Open Camera"
      : mode === "camera"
        ? isRecording
          ? "Stop & Translate"
          : "Start Capture"
        : "Select Video Above";

  const primaryActionDisabled =
    mode === "video" ||
    translation.isProcessing ||
    (mode === "camera" && (!isActive || isStarting));

  return (
    <div className="translate-page">
      <div className="translate-glow translate-glow-left" />
      <div className="translate-glow translate-glow-right" />

      <section className="translate-hero-panel">
        <div className="translate-hero-copy">
          <span className="translate-kicker">
            real-time sign language assistant
          </span>
          <h1 className={titleVariant}>{displayTitle}</h1>
          <p className="translate-subtitle">{heroSubtitle}</p>
        </div>

        <div className="translate-summary-card">
          <div className="summary-card-header">
            <div className="summary-header-left">
              <span className="summary-pill">Current mode</span>
              <span
                className={`summary-live-badge ${translation.isProcessing ? "processing" : isRecording ? "recording" : ""}`}
              >
                <span className="summary-live-dot" />
                {liveStatusLabel}
              </span>
            </div>
            <span className="summary-mode-value">{mode}</span>
          </div>

          <div className="summary-card-body">
            <div className="summary-main-text">{summaryText}</div>

            <div className="summary-meta-grid">
              <div className="summary-meta-item">
                <span className="summary-meta-label">状态</span>
                <strong>
                  {translation.isProcessing
                    ? "处理中"
                    : isRecording
                      ? "录制中"
                      : isActive
                        ? "就绪"
                        : error
                          ? "异常"
                          : "待命"}
                </strong>
              </div>
              <div className="summary-meta-item">
                <span className="summary-meta-label">置信度</span>
                <strong>
                  {latestResult
                    ? `${(latestResult.confidence * 100).toFixed(1)}%`
                    : "--"}
                </strong>
              </div>
              <div className="summary-meta-item">
                <span className="summary-meta-label">最新时间</span>
                <strong>
                  {latestResult
                    ? latestResult.timestamp.toLocaleTimeString("zh-CN")
                    : "--:--:--"}
                </strong>
              </div>
              <div className="summary-meta-item">
                <span className="summary-meta-label">摄像头</span>
                <strong>
                  {error
                    ? "需重试"
                    : isStarting
                      ? "启动中"
                      : isActive
                        ? "已连接"
                        : "未连接"}
                </strong>
              </div>
              <div className="summary-meta-item">
                <span className="summary-meta-label">推理来源</span>
                <strong>
                  {latestResult
                    ? latestResult.source === "backend"
                      ? "真实后端"
                      : "错误返回"
                    : "--"}
                </strong>
              </div>
              <div className="summary-meta-item">
                <span className="summary-meta-label">有效帧数</span>
                <strong>{latestResult?.frames ?? "--"}</strong>
              </div>
            </div>
          </div>
        </div>

        <ResultPanel
          results={translation.results}
          onRemove={translation.removeResult}
          onSpeak={voice.speak}
        />
      </section>

      <section className="translate-studio-panel">
        <div className="studio-card">
          <div className="studio-window-bar">
            <span className="studio-window-dot" />
            <span className="studio-window-dot" />
            <span className="studio-window-dot" />
          </div>

          <Sidebar
            currentMode={mode}
            onModeChange={handleModeChange}
            isRecording={isRecording}
            isCameraActive={isActive}
            isProcessing={translation.isProcessing}
          />

          <div className="studio-preview-shell">
            {mode === "camera" && (
              <CameraView
                mode={mode}
                isCameraActive={isActive}
                isStarting={isStarting}
                isRecording={isRecording}
                frameCount={frameCount}
                error={error}
                onVideoRef={handleVideoRef}
              />
            )}

            {mode === "video" && (
              <VideoUpload
                isProcessing={translation.isProcessing}
                onFileSelected={handleFileSelected}
              />
            )}

            {mode === "help" && <HelpPanel />}

            {translation.isProcessing && mode === "video" && (
              <div className="processing-overlay">
                <div className="processing-spinner" />
              </div>
            )}
          </div>

          <p className="studio-disclaimer">
            Video recording and user data will be subject to these terms upon
            submission.
          </p>

          <div className="studio-actions">
            <button
              className={`studio-primary-btn ${isRecording ? "recording" : ""}`}
              type="button"
              onClick={handlePrimaryAction}
              disabled={primaryActionDisabled}
            >
              <span className="studio-primary-btn-icon">
                {translation.isProcessing ? "◌" : isRecording ? "■" : "↻"}
              </span>
              {primaryActionLabel}
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}
