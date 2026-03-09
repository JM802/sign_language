import { useState, useRef } from "react";
import type { TranslationResult } from "../hooks/useTranslation";

interface ResultBubbleProps {
  result: TranslationResult;
  onRemove: (id: string) => void;
  onSpeak: (text: string) => void;
}

/**
 * 结果气泡组件 - 对应 PyQt 中的 ResultBubble
 * 渐变背景、hover 展开、点击朗读、可关闭、关闭时向上淡出
 */
export default function ResultBubble({
  result,
  onRemove,
  onSpeak,
}: ResultBubbleProps) {
  const [isHovered, setIsHovered] = useState(false);
  const [isClosing, setIsClosing] = useState(false);
  const bubbleRef = useRef<HTMLDivElement>(null);

  const handleClose = (e: React.MouseEvent) => {
    e.stopPropagation();
    setIsClosing(true);
    // 等动画结束后再真的删除
    setTimeout(() => onRemove(result.id), 450);
  };

  const handleClick = () => {
    onSpeak(result.text);
  };

  return (
    <div
      ref={bubbleRef}
      className={`result-bubble ${isHovered ? "expanded" : ""} ${isClosing ? "closing" : ""}`}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onClick={handleClick}
      title="点击朗读"
    >
      <button className="bubble-close-btn" onClick={handleClose} title="删除">
        ×
      </button>

      <div className="bubble-content">
        <div className="bubble-text">{result.text}</div>
        <div className="bubble-meta">
          <span className="bubble-confidence">
            置信度: {(result.confidence * 100).toFixed(1)}%
          </span>
          <span className="bubble-time">
            {result.timestamp.toLocaleTimeString("zh-CN")}
          </span>
        </div>
      </div>

      <div className="bubble-speak-hint">🔊 点击朗读</div>
    </div>
  );
}
