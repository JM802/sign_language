import { useRef, useState, useCallback, useEffect } from "react";

/**
 * 摄像头管理 Hook - 对应 PyQt 中的 CameraThread
 * 使用 getUserMedia 打开浏览器摄像头
 */
export function useCamera() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const recordingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(
    null,
  );

  const [isActive, setIsActive] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [frameCount, setFrameCount] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const frameCountRef = useRef(0);

  // 开启摄像头
  const startCamera = useCallback(async (videoElement: HTMLVideoElement) => {
    try {
      if (streamRef.current) {
        videoElement.srcObject = streamRef.current;
        await videoElement.play();
        videoRef.current = videoElement;
        setIsActive(true);
        setError(null);
        return;
      }

      setIsStarting(true);
      setError(null);

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          aspectRatio: { ideal: 16 / 9 },
          facingMode: "user",
        },
        audio: false,
      });
      videoElement.srcObject = stream;
      await videoElement.play();
      videoRef.current = videoElement;
      streamRef.current = stream;
      setIsActive(true);
      setError(null);
    } catch (err) {
      console.error("摄像头启动失败:", err);
      setIsActive(false);
      setError("摄像头启动失败，请检查浏览器权限后重试。");
      throw err;
    } finally {
      setIsStarting(false);
    }
  }, []);

  // 关闭摄像头
  const stopCamera = useCallback(() => {
    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state !== "inactive"
    ) {
      mediaRecorderRef.current.stop();
    }
    if (recordingIntervalRef.current) {
      clearInterval(recordingIntervalRef.current);
      recordingIntervalRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
      videoRef.current = null;
    }
    setIsActive(false);
    setIsRecording(false);
    setFrameCount(0);
    frameCountRef.current = 0;
  }, []);

  // 开始/停止录制 - 对应 PyQt CameraThread.toggle_recording
  const toggleRecording = useCallback((): Promise<Blob | null> => {
    return new Promise((resolve) => {
      if (!isRecording) {
        // 开始录制
        if (!streamRef.current) {
          resolve(null);
          return;
        }

        chunksRef.current = [];
        frameCountRef.current = 0;
        setFrameCount(0);

        const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
          ? "video/webm;codecs=vp9"
          : MediaRecorder.isTypeSupported("video/webm;codecs=vp8")
            ? "video/webm;codecs=vp8"
            : "video/webm";

        const recorder = new MediaRecorder(streamRef.current, {
          mimeType,
        });

        recorder.ondataavailable = (e) => {
          if (e.data.size > 0) {
            chunksRef.current.push(e.data);
          }
        };

        recorder.onstop = () => {
          const blob = new Blob(chunksRef.current, { type: "video/webm" });
          resolve(blob);
        };

        recorder.start(100); // 每 100ms 收集一段
        mediaRecorderRef.current = recorder;
        setIsRecording(true);

        // 帧计数器
        const countInterval = setInterval(() => {
          frameCountRef.current += 3;
          setFrameCount(frameCountRef.current);
        }, 100);

        recordingIntervalRef.current = countInterval;

        // 这次 toggle 是"开始"，不返回数据
        resolve(null);
      } else {
        // 停止录制
        if (
          mediaRecorderRef.current &&
          mediaRecorderRef.current.state !== "inactive"
        ) {
          if (recordingIntervalRef.current) {
            clearInterval(recordingIntervalRef.current);
            recordingIntervalRef.current = null;
          }
          mediaRecorderRef.current.stop(); // 触发 onstop -> resolve
        } else {
          resolve(null);
        }
        setIsRecording(false);
        setFrameCount(0);
        frameCountRef.current = 0;
      }
    });
  }, [isRecording]);

  // 清理
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  return {
    isActive,
    isStarting,
    isRecording,
    frameCount,
    error,
    startCamera,
    stopCamera,
    toggleRecording,
  };
}
