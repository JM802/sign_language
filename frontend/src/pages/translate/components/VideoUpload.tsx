import { useRef } from "react";

interface VideoUploadProps {
  isProcessing: boolean;
  onFileSelected: (file: File) => void;
}

/**
 * 视频上传组件 - 对应 PyQt 中 handle_video_import 的文件选择功能
 */
export default function VideoUpload({
  isProcessing,
  onFileSelected,
}: VideoUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleClick = () => {
    inputRef.current?.click();
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onFileSelected(file);
      // 重置 input 以便再次选择同一文件
      e.target.value = "";
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("video/")) {
      onFileSelected(file);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  return (
    <div className="video-upload-area">
      <input
        ref={inputRef}
        type="file"
        accept="video/mp4,video/avi,video/mkv,video/webm"
        onChange={handleChange}
        style={{ display: "none" }}
      />

      <div
        className={`video-upload-dropzone ${isProcessing ? "processing" : ""}`}
        onClick={handleClick}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        {isProcessing ? (
          <>
            <div className="upload-spinner" />
            <div className="upload-text">正在分析视频，请稍候...</div>
          </>
        ) : (
          <>
            <div className="upload-icon">📁</div>
            <div className="upload-text">
              点击选择视频文件
              <br />
              <span className="upload-hint">或拖拽文件到此处</span>
            </div>
            <div className="upload-formats">支持格式: MP4, AVI, MKV, WebM</div>
          </>
        )}
      </div>
    </div>
  );
}
