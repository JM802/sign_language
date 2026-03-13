declare module '*.mp4' {
    const src: string;
    export default src;
}

// 顺便把其他常见静态资源也加上
declare module '*.mov' {
    const src: string;
    export default src;
}

declare module '*.webm' {
    const src: string;
    export default src;
}