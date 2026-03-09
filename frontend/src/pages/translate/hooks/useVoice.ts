import { useCallback } from "react";

/**
 * 语音播报 Hook - 对应 PyQt 中的 VoiceAssistant
 * 使用浏览器原生 Web Speech API 实现 TTS
 */
export function useVoice() {
  const speak = useCallback((text: string) => {
    if (!("speechSynthesis" in window)) {
      console.warn("当前浏览器不支持语音合成");
      return;
    }

    // 清空队列，只播报最新内容
    window.speechSynthesis.cancel();

    // 清理文本
    const clean = text
      .replace(/识别结果：/g, "")
      .replace(/识别成功：/g, "")
      .replace(/从图片提取的文字：/g, "")
      .replace(/\([^)]*置信度[^)]*\)/g, "")
      .replace(/\n/g, "，")
      .trim();

    if (!clean) return;

    const utterance = new SpeechSynthesisUtterance(clean);
    utterance.lang = "zh-CN";
    utterance.rate = 1.0;
    utterance.volume = 1.0;

    // 优先选中文语音
    const voices = window.speechSynthesis.getVoices();
    const zhVoice = voices.find(
      (v) => v.lang.includes("zh") || v.name.includes("Chinese"),
    );
    if (zhVoice) {
      utterance.voice = zhVoice;
    }

    window.speechSynthesis.speak(utterance);
  }, []);

  const stop = useCallback(() => {
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
  }, []);

  return { speak, stop };
}
