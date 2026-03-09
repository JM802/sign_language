import { useState, useCallback } from "react";

const API_BASE =
  import.meta.env.VITE_API_BASE ||
  `${window.location.protocol}//${window.location.hostname}:8000`;

export interface TranslationResult {
  id: string;
  text: string;
  rawText?: string;
  confidence: number;
  timestamp: Date;
  frames?: number;
  source: "backend" | "error";
}

/**
 * 翻译推理 Hook - 对应 PyQt 中的 model_engine + camera_manager 的推理部分
 * 将视频/录制数据发送到后端 API 进行推理
 */
export function useTranslation() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState<TranslationResult[]>([]);

  const appendResult = useCallback((result: TranslationResult) => {
    setResults((prev) => [...prev, result]);
    return result;
  }, []);

  // 发送视频 blob 到后端推理
  const inferFromBlob = useCallback(
    async (blob: Blob): Promise<TranslationResult | null> => {
      setIsProcessing(true);
      try {
        const formData = new FormData();
        formData.append("video", blob, "recording.webm");

        const res = await fetch(`${API_BASE}/api/inference`, {
          method: "POST",
          body: formData,
        });

        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          throw new Error(data.detail || `推理请求失败: ${res.status}`);
        }

        const result: TranslationResult = {
          id: crypto.randomUUID(),
          text: data.text || "未能识别",
          rawText: data.raw_text,
          confidence: data.confidence || 0,
          timestamp: new Date(),
          frames: data.frames,
          source: "backend",
        };

        return appendResult(result);
      } catch (err) {
        console.error("推理失败:", err);

        const errorResult: TranslationResult = {
          id: crypto.randomUUID(),
          text:
            err instanceof Error
              ? `识别失败：${err.message}`
              : "识别失败：无法连接到后端推理服务。",
          confidence: 0,
          timestamp: new Date(),
          source: "error",
        };
        return appendResult(errorResult);
      } finally {
        setIsProcessing(false);
      }
    },
    [appendResult],
  );

  // 上传视频文件推理
  const inferFromFile = useCallback(
    async (file: File): Promise<TranslationResult | null> => {
      setIsProcessing(true);
      try {
        const formData = new FormData();
        formData.append("video", file);

        const res = await fetch(`${API_BASE}/api/inference`, {
          method: "POST",
          body: formData,
        });

        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          throw new Error(data.detail || `推理请求失败: ${res.status}`);
        }

        const result: TranslationResult = {
          id: crypto.randomUUID(),
          text: data.text || "未能识别",
          rawText: data.raw_text,
          confidence: data.confidence || 0,
          timestamp: new Date(),
          frames: data.frames,
          source: "backend",
        };

        return appendResult(result);
      } catch (err) {
        console.error("推理失败:", err);

        const errorResult: TranslationResult = {
          id: crypto.randomUUID(),
          text:
            err instanceof Error
              ? `识别失败：${err.message}`
              : `识别失败：无法处理文件 ${file.name}`,
          confidence: 0,
          timestamp: new Date(),
          source: "error",
        };
        return appendResult(errorResult);
      } finally {
        setIsProcessing(false);
      }
    },
    [appendResult],
  );

  // 移除某条结果
  const removeResult = useCallback((id: string) => {
    setResults((prev) => prev.filter((r) => r.id !== id));
  }, []);

  // 清空所有结果
  const clearResults = useCallback(() => {
    setResults([]);
  }, []);

  return {
    isProcessing,
    results,
    inferFromBlob,
    inferFromFile,
    removeResult,
    clearResults,
  };
}
