import { useState } from 'react';
import '../../styles/contact.css';
// 假设这是你的地图背景图，如果没有，暂时用纯色代替
import MapBg from '../../assets/image/map.png';

// 这里为了方便演示直接用了SVG，你可以换成 react-icons
const GithubIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path></svg>
);
const MailIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"></path><polyline points="22,6 12,13 2,6"></polyline></svg>
);
const PinIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path><circle cx="12" cy="10" r="3"></circle></svg>
);

export default function Contact() {
    // 记录当前哪一个 item 被点击展开了
    const [activeId, setActiveId] = useState<number | null>(null);

    const contactData = [
        {
            id: 1,
            icon: <GithubIcon />,
            label: "Github",
            content: "github.com/JM802/sign_language",
            link: "https://github.com/JM802/sign_language"
        },
        {
            id: 2,
            icon: <MailIcon />,
            label: "Email",
            content: "hello@studio.com",
            link: "mailto:hello@studio.com"
        },
        {
            id: 3,
            icon: <PinIcon />,
            label: "Location",
            content: "Wuhan, China",
            link: "#"
        }
    ];

    const toggleBubble = (id: number) => {
        if (activeId === id) {
            setActiveId(null); // 如果已经点开，再次点击则关闭
        } else {
            setActiveId(id);
        }
    };

    return (
        <div className="contact-container">
            {/* 左侧地图区域 */}
            <div className="map-section" style={{ backgroundImage: `url(${MapBg})` }}>
                <div className="map-overlay"></div>
                {/* 模拟地图上的定位点呼吸动画 */}
                <div className="map-marker">
                    <div className="pulse-ring"></div>
                    <div className="dot"></div>
                    <div className="tooltip">We are here!</div>
                </div>
            </div>

            {/* 右侧交互区域 */}
            <div className="info-section">
                <div className="info-content">
                    <h1 className="title">
                        {['G','e','t',' ','i','n',' ','T','o','u','c','h'].map((char, i) => (
                            <span
                                key={i}
                                className="letter"
                                style={{ animationDelay: `${(i + 1) * 0.05}s` }}
                            >
      {char === ' ' ? '\u00A0' : char}
    </span>
                        ))}
                    </h1>
                    <p className="subtitle">Let's create something specific.</p>

                    <div className="icons-grid">
                        {contactData.map((item) => (
                            <div
                                key={item.id}
                                className={`icon-wrapper ${activeId === item.id ? 'active' : ''}`}
                                onClick={() => toggleBubble(item.id)}
                            >
                                <div className="icon-btn">
                                    {item.icon}
                                </div>

                                {/* 气泡框 */}
                                <div className={`bubble ${activeId === item.id ? 'show' : ''}`}>
                                    <span className="bubble-label">{item.label}</span>
                                    <a href={item.link} target="_blank" rel="noreferrer" className="bubble-link">
                                        {item.content}
                                    </a>
                                    <div className="bubble-arrow"></div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}