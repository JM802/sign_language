import React, { useRef } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { useGSAP } from '@gsap/react';
import '../../styles/about.css'

gsap.registerPlugin(ScrollTrigger);

interface HistoryNode {
    id: number;
    year: string;
    title: string;
    description: string;
    imageUrl: string;
}

const aboutData: HistoryNode[] = [
    {
        id: 1,
        year: "1",
        title: "Translate Page",
        description: "打开摄像头之后，您可以做出手语动作，然后左边部分会反馈以模型的文本输出、置信度和对应的SRHand教学动作",
        imageUrl: "https://picsum.photos/800/600?random=1" // 👈 换成这个
    },
    {
        id: 2,
        year: "2",
        title: "Learning Page",
        description: "这里有许多生动的卡片，鼠标悬停之上可以自动播放教学视频",
        imageUrl: "https://picsum.photos/800/600?random=2" // 👈 换成这个
    },
    {
        id: 3,
        year: "3",
        title: "Contact Page",
        description: "在这里你可以了解我们的信息，期待与您的进一步交流！",
        imageUrl: "https://picsum.photos/800/600?random=3" // 👈 换成这个
    },
];
const About: React.FC = () => {
    const containerRef = useRef<HTMLDivElement>(null);

    useGSAP(() => {
        const items = gsap.utils.toArray<HTMLElement>('.about-item');

        items.forEach((item) => {
            gsap.fromTo(item,
                {
                    opacity: 0,
                    y: 100 // 初始状态在下方 100px 的位置
                },
                {
                    opacity: 1,
                    y: 0,  // 升到原本的位置
                    duration: 1,
                    ease: "power3.out",
                    scrollTrigger: {
                        trigger: item,
                        // 核心逻辑：当这个 item 的顶部，到达屏幕视口高度的 85% 位置时，开始播放动画
                        start: "top 85%",
                        // 往下滚播放，往上滚隐藏（如果要只播放一次，把这里的 reverse 改为 none）
                        toggleActions: "play none none reverse",
                        // markers: true // 如果你想看触发的红绿线，可以取消这行注释
                    }
                }
            );
        });
    }, { scope: containerRef });

    return (
        <section className="about-section" ref={containerRef}>
            <div className="about-container">
                <div className="about-header">
                    <h2>About How to Use It</h2>
                    <p>The journey of our web</p>
                </div>

                <div className="about-line"></div>

                {aboutData.map((node, index) => {
                    const isReverse = index % 2 !== 0;

                    return (
                        <div key={node.id} className={`about-item ${isReverse ? 'reverse' : ''}`}>
                            <div className="about-content">
                                <h3>{node.title}</h3>
                                <p>{node.description}</p>
                            </div>
                            <div className="about-marker">
                                {node.year}
                            </div>
                            <div className="about-image-wrapper">
                                <img src={node.imageUrl} alt={node.title} />
                            </div>
                        </div>
                    );
                })}
            </div>
        </section>
    );
};

export default About;