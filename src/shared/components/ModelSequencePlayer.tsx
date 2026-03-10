import React, { useRef, Suspense, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Stage, useGLTF, Html, useProgress } from '@react-three/drei';
import * as THREE from 'three';
import gsap from 'gsap';
import { useGSAP } from '@gsap/react';

// --- 加载进度条组件 ---
const Loader = () => {
    const { progress } = useProgress();
    return (
        <Html center>
            <div style={{ color: 'white', whiteSpace: 'nowrap' }}>
                Loading Models... {progress.toFixed(0)}%
            </div>
        </Html>
    );
};

// --- 子组件：负责循环控制 (直接操作 DOM/Three.js 属性，脱离 React State) ---
interface SequenceProps {
    urls: string[];
    fps?: number; // 播放帧率
}

const SequenceController: React.FC<SequenceProps> = ({ urls, fps = 12 }) => {
    // 1. 一次性并行加载所有模型序列
    // @ts-ignore (因为 useGLTF 类型定义有时会报数组错误，但实际完全支持数组)
    const gltfs = useGLTF(urls) as any[];

    const groupRef = useRef<THREE.Group>(null);
    const clockRef = useRef(0);
    const currentIndexRef = useRef(0);

    // 2. 初始化：让第一个模型显示，其他的隐藏
    useEffect(() => {
        if (groupRef.current) {
            groupRef.current.children.forEach((child, index) => {
                child.visible = index === 0;
            });
        }
    }, [gltfs]);

    // 3. 高性能动画循环 (不触发组件 re-render)
    useFrame((_, delta) => {
        if (!groupRef.current || gltfs.length === 0) return;

        clockRef.current += delta;

        // 如果累计的时间超过了 1 帧所需的时间
        if (clockRef.current > 1 / fps) {
            const children = groupRef.current.children;

            // 隐藏当前帧
            if (children[currentIndexRef.current]) {
                children[currentIndexRef.current].visible = false;
            }

            // 计算下一帧的索引
            currentIndexRef.current = (currentIndexRef.current + 1) % gltfs.length;

            // 显示下一帧
            if (children[currentIndexRef.current]) {
                children[currentIndexRef.current].visible = true;
            }

            // 减去一帧的时间，保证动画平滑不丢失精度
            clockRef.current -= 1 / fps;
        }
    });

    return (
        <group ref={groupRef}>
            {gltfs.map((gltf, index) => (
                // 默认仅让第一帧可见，后续交由 useFrame 控制
                <primitive key={urls[index]} object={gltf.scene} visible={index === 0} />
            ))}
        </group>
    );
};

// 1. 在 Props 里增加 fps (帧率) 属性，用于控制快慢
interface ModelSequencePlayerProps {
    isStreaming: boolean;
    glbUrls: string[];
    fps?: number; // 新增：控制播放速度
}

const ModelSequencePlayer: React.FC<ModelSequencePlayerProps> = ({
                                                                     isStreaming,
                                                                     glbUrls,
                                                                     fps = 5
                                                                 }) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const canvasRef = useRef<HTMLDivElement>(null);

    // 预加载所有模型 (即使关闭摄像头，模型依然在内存里，下次打开秒开)
    useEffect(() => {
        if (glbUrls.length > 0) {
            useGLTF.preload(glbUrls);
        }
    }, [glbUrls]);

    // 使用 GSAP 每次打开摄像头时重新触发淡入效果
    useGSAP(() => {
        if (isStreaming && glbUrls.length > 0 && canvasRef.current) {
            gsap.fromTo(canvasRef.current,
                { opacity: 0 },
                { opacity: 1, duration: 1.5, ease: 'power2.out' }
            );
        }
    }, [isStreaming, glbUrls.length]);

    return (
        <div className="ai-output-wrapper" ref={containerRef}>
            <div className="output-3d-container">

                {/* 修改：当没在推流时，显示提示文案 */}
                {!isStreaming && (
                    <div className="empty-state">Camera is off. Translation stopped.</div>
                )}

                {isStreaming && glbUrls.length === 0 && (
                    <div className="loading-state">Generating 3D Models...</div>
                )}

                {/* 核心修改：只有 isStreaming 为 true 且有数据时，才渲染 Canvas */}
                {isStreaming && glbUrls.length > 0 && (
                    <div className="canvas-wrapper" ref={canvasRef}>
                        <Canvas camera={{ position: [0, 2, 5], fov: 50 }}>
                            <ambientLight intensity={0.5} />
                            <directionalLight position={[10, 10, 5]} intensity={1} />

                            <Suspense fallback={<Loader />}>
                                <Stage environment="city" intensity={0.6}>
                                    <SequenceController urls={glbUrls} fps={fps} />
                                </Stage>
                            </Suspense>

                            <OrbitControls autoRotate={false} />
                        </Canvas>
                    </div>
                )}

            </div>
        </div>
    );
};

export default ModelSequencePlayer;