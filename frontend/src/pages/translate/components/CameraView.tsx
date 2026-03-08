import { useRef, useEffect } from "react";
import type { TranslateMode } from "./Sidebar";

interface CameraViewProps {
  mode: TranslateMode;
  isCameraActive: boolean;
  isStarting: boolean;
  isRecording: boolean;
  frameCount: number;
  error: string | null;
  onVideoRef: (el: HTMLVideoElement) => void;
}

/**
 * 摄像头视图 - 对应 PyQt 中 image_label 显示摄像头画面
 */
export default function CameraView({
  mode,
  isCameraActive,
  isStarting,
  isRecording,
  frameCount,
  error,
  onVideoRef,
}: CameraViewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (videoRef.current && mode === "camera") {
      onVideoRef(videoRef.current);
    }
  }, [mode, onVideoRef]);

  if (mode !== "camera") return null;

  return (
    <div className="camera-view">
      <video
        ref={videoRef}
        className="camera-video"
        autoPlay
        playsInline
        muted
        style={{ transform: "scaleX(-1)" }} // 镜像
      />

      {/* 录制状态指示器 - 对应 PyQt 中 cv2.putText 的 "REC" */}
      {isRecording && (
        <div className="camera-recording-indicator">
          <span className="rec-dot" />
          <span>REC {frameCount}</span>
        </div>
      )}

      {/* 摄像头未开启时的提示 */}
      {!isCameraActive && (
        <div className="camera-placeholder">
          <div className="camera-placeholder-icon">📷</div>
          <div>
            {isStarting ? "正在打开摄像头..." : (error ?? "摄像头未连接")}
          </div>
          {!isStarting && (
            <button
              type="button"
              className="camera-retry-btn"
              onClick={() => {
                if (videoRef.current) {
                  void onVideoRef(videoRef.current);
                }
              }}
            >
              重试启动
            </button>
          )}
        </div>
      )}
    </div>
  );
}
