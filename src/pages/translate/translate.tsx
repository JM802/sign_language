import { useState, useEffect } from 'react';
import Webcam from 'react-webcam';
import ModelSequencePlayer from '../../shared/components/ModelSequencePlayer.tsx'
import '../../styles/translate.css'

export default function translate() {
    const [isStreaming, setIsStreaming] = useState(false);
    const [translatedText, setTranslatedText] = useState("Waiting for input...");
    const [confidence, setConfidence] = useState(0);
    const [glbFrames] = useState<string[]>([
        '/models/frame_000.glb',
        '/models/frame_001.glb',
        '/models/frame_002.glb',
        '/models/frame_003.glb',
        '/models/frame_004.glb',
        '/models/frame_005.glb',
        '/models/frame_006.glb',
        '/models/frame_007.glb',
        '/models/frame_008.glb',
        '/models/frame_009.glb',
        '/models/frame_010.glb',
        '/models/frame_011.glb',
        '/models/frame_012.glb',
        '/models/frame_013.glb',
        '/models/frame_014.glb',
        '/models/frame_015.glb',
        '/models/frame_016.glb',
        '/models/frame_017.glb',
        '/models/frame_018.glb',
        '/models/frame_019.glb',
        '/models/frame_020.glb',
        '/models/frame_021.glb',
        '/models/frame_022.glb',
        '/models/frame_023.glb',
        '/models/frame_024.glb',
        '/models/frame_025.glb',
        '/models/frame_026.glb',
        '/models/frame_027.glb',
        '/models/frame_028.glb',
    ]);

    // 模拟 AI 翻译过程
    useEffect(() => {
        if (!isStreaming) {
            setTranslatedText("Camera is off.");
            setConfidence(0);
            return;
        }

        const phrases = ["Hello", "Welcome", "Sign Language", "AI Translation", "Good Morning"];
        const interval = setInterval(() => {
            // 模拟随机识别
            const randomText = phrases[Math.floor(Math.random() * phrases.length)];
            setTranslatedText(randomText);
            setConfidence(Math.floor(Math.random() * 20) + 80); // 80% - 99%
        }, 2000);

        return () => clearInterval(interval);
    }, [isStreaming]);

    return (
        <div className="sign-container">
            {/* 中间的大卡片 */}
            <div className="sign-card">

                {/* 左侧：翻译结果展示 (Result) */}
                <div className="text-section">
                    <div className="text-header">
                        <span className="live-dot"></span>
                        {isStreaming ? "LIVE TRANSLATION" : "OFFLINE"}
                    </div>

                    <div className="text-display">
                        <h1 className="main-text">{translatedText}</h1>
                        {isStreaming && (
                            <div className="confidence-badge">
                                Confidence: {confidence}%
                            </div>
                        )}
                    </div>

                    <ModelSequencePlayer
                        isStreaming={isStreaming}
                        glbUrls={glbFrames}
                    />

                    <div className="instruction">
                        <p>Perform sign language gestures in front of the camera.</p>
                    </div>
                </div>


                {/* 右侧：摄像头输入 (Input) */}
                <div className="camera-section">
                    <div className="camera-wrapper">
                        {isStreaming ? (
                            <>
                                <Webcam
                                    audio={false}
                                    className="webcam-video"
                                    videoConstraints={{ facingMode: "user" }}
                                />
                                {/* 模拟 AI 骨架识别层 */}
                                <div className="skeleton-overlay">
                                    <HandSkeletonSVG />
                                </div>
                            </>
                        ) : (
                            <div className="camera-placeholder">
                                <div className="placeholder-icon">📷</div>
                            </div>
                        )}
                    </div>

                    {/* 悬浮控制栏 */}
                    <div className="control-bar">
                        <button
                            className={`toggle-btn ${isStreaming ? 'active' : ''}`}
                            onClick={() => setIsStreaming(!isStreaming)}
                        >
                            {isStreaming ? "Stop Translation" : "Start Camera"}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}

// 一个简单的 SVG 组件，模拟手部骨架连接点
const HandSkeletonSVG = () => (
    <svg viewBox="0 0 200 200" className="hand-svg">
        {/* 手掌中心 */}
        <circle cx="100" cy="150" r="4" fill="#00f0ff" />

        {/* 大拇指 */}
        <path d="M100 150 L60 140 L40 120 L30 100" stroke="rgba(0, 240, 255, 0.6)" strokeWidth="2" fill="none"/>
        <circle cx="60" cy="140" r="3" fill="#00f0ff" />
        <circle cx="40" cy="120" r="3" fill="#00f0ff" />
        <circle cx="30" cy="100" r="3" fill="#00f0ff" />

        {/* 食指 */}
        <path d="M100 150 L90 110 L85 80 L80 50" stroke="rgba(0, 240, 255, 0.6)" strokeWidth="2" fill="none"/>
        <circle cx="90" cy="110" r="3" fill="#00f0ff" />
        <circle cx="85" cy="80" r="3" fill="#00f0ff" />
        <circle cx="80" cy="50" r="3" fill="#00f0ff" />

        {/* 中指 */}
        <path d="M100 150 L105 105 L108 70 L110 40" stroke="rgba(0, 240, 255, 0.6)" strokeWidth="2" fill="none"/>
        <circle cx="105" cy="105" r="3" fill="#00f0ff" />
        <circle cx="108" cy="70" r="3" fill="#00f0ff" />
        <circle cx="110" cy="40" r="3" fill="#00f0ff" />

        {/* 动画光圈 */}
        <circle cx="100" cy="150" r="10" stroke="#00f0ff" strokeWidth="1" fill="none" className="pulse-circle"/>
    </svg>
);