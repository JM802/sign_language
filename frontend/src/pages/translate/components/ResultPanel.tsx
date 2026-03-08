import { useRef, useEffect } from "react";
import ResultBubble from "./ResultBubble";
import type { TranslationResult } from "../hooks/useTranslation";

interface ResultPanelProps {
  results: TranslationResult[];
  onRemove: (id: string) => void;
  onSpeak: (text: string) => void;
}

/**
 * 右侧结果面板 - 对应 PyQt 中 scrollArea + result_layout
 * 可滚动显示所有识别结果气泡
 */
export default function ResultPanel({
  results,
  onRemove,
  onSpeak,
}: ResultPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // 新结果出现时自动滚到底部
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [results.length]);

  return (
    <div className="result-panel">
      <div className="result-panel-header">
        <span className="result-panel-icon">💬</span>
        <span>历史识别结果</span>
      </div>

      <div className="result-panel-scroll" ref={scrollRef}>
        {results.length === 0 ? (
          <div className="result-empty">
            <div className="result-empty-icon">📝</div>
            <div>暂无识别结果</div>
            <div className="result-empty-hint">
              使用摄像头录制手语动作
              <br />
              或上传视频文件开始识别
            </div>
          </div>
        ) : (
          results.map((r) => (
            <ResultBubble
              key={r.id}
              result={r}
              onRemove={onRemove}
              onSpeak={onSpeak}
            />
          ))
        )}
      </div>
    </div>
  );
}
