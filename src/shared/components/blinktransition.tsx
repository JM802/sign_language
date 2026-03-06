import React, { createContext, useContext, useRef, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import gsap from "gsap";
import { useGSAP } from "@gsap/react";

gsap.registerPlugin(useGSAP);

interface TransitionContextType {
    navigateWithTransition: (to: string) => void;
}

const TransitionContext = createContext<TransitionContextType>({
    navigateWithTransition: () => {}
});

export const BlinkTransitionProvider: React.FC<{ children: React.ReactNode }> = ({
                                                                                     children
                                                                                 }) => {
    const navigate = useNavigate();
    const location = useLocation();

    const containerRef = useRef<HTMLDivElement>(null);
    const topEyelidRef = useRef<HTMLDivElement>(null);
    const bottomEyelidRef = useRef<HTMLDivElement>(null);

    const timelineRef = useRef<gsap.core.Timeline | null>(null);

    const pendingRouteRef = useRef<string | null>(null);
    const isAnimatingRef = useRef(false);

    const [ready, setReady] = useState(false);

    useGSAP(
        () => {
            const ctx = gsap.context(() => {
                const tl = gsap.timeline({
                    paused: true,
                    onStart: () => {
                        isAnimatingRef.current = true;
                        if (containerRef.current)
                            containerRef.current.style.pointerEvents = "all";
                    },
                    onComplete: () => {
                        isAnimatingRef.current = false;
                        if (containerRef.current)
                            containerRef.current.style.pointerEvents = "none";
                    }
                });

                tl.to([topEyelidRef.current, bottomEyelidRef.current], {
                    scaleY: 1,
                    duration: 0.4,
                    ease: "power3.in"
                })

                    .addLabel("switch")

                    .call(() => {
                        if (pendingRouteRef.current) {
                            navigate(pendingRouteRef.current);
                            window.scrollTo(0, 0);
                            pendingRouteRef.current = null;
                        }
                    })

                    .to([topEyelidRef.current, bottomEyelidRef.current], {
                        scaleY: 0,
                        duration: 0.4,
                        ease: "power2.out",
                        delay: 0.1
                    });

                timelineRef.current = tl;
                setReady(true);

                // 首次加载自动眨眼
                tl.play();
            }, containerRef);

            return () => ctx.revert();
        },
        { scope: containerRef }
    );

    const navigateWithTransition = (to: string) => {
        if (!ready) return;
        if (location.pathname === to) return;
        if (isAnimatingRef.current) return;

        pendingRouteRef.current = to;

        timelineRef.current?.restart();
    };

    return (
        <TransitionContext.Provider value={{ navigateWithTransition }}>
            {children}

            {/* 转场遮罩 */}
            <div ref={containerRef} style={overlayStyle}>
                <div
                    ref={topEyelidRef}
                    style={{ ...eyelidStyle, transformOrigin: "top" }}
                />

                <div
                    ref={bottomEyelidRef}
                    style={{ ...eyelidStyle, transformOrigin: "bottom" }}
                />
            </div>
        </TransitionContext.Provider>
    );
};

export const useBlinkNavigate = () => {
    return useContext(TransitionContext).navigateWithTransition;
};

interface BlinkLinkProps
    extends React.AnchorHTMLAttributes<HTMLAnchorElement> {
    to: string;
    children: React.ReactNode;
}

export const BlinkLink: React.FC<BlinkLinkProps> = ({
                                                        to,
                                                        children,
                                                        className,
                                                        style
                                                    }) => {
    const blinkNavigate = useBlinkNavigate();

    const handleClick = (e: React.MouseEvent) => {
        e.preventDefault();
        blinkNavigate(to);
    };

    return (
        <a
            href={to}
            onClick={handleClick}
            className={className}
            style={{ cursor: "pointer", ...style }}
        >
            {children}
        </a>
    );
};

const overlayStyle: React.CSSProperties = {
    position: "fixed",
    inset: 0,
    width: "100vw",
    height: "100vh",
    pointerEvents: "none",
    zIndex: 9999,
    display: "flex",
    flexDirection: "column"
};

const eyelidStyle: React.CSSProperties = {
    width: "100%",
    height: "50%",
    backgroundColor: "#1a1a1a",
    transform: "scaleY(0)"
};