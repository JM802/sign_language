import Banana from '../../assets/image/banana.png'
import Hello from '../../assets/image/hello.png'
import Baby from '../../assets/image/baby.png'
import Bread from '../../assets/image/bread.png'
import Halloween from '../../assets/image/halloween.png'

import banana from '../../assets/video/banana.mp4'


import '../../styles/learning.css'
import {useRef} from "react";

export default function Learning() {
    const sliderRef = useRef<HTMLDivElement>(null);

    // 下一张的逻辑
    const handleNext = () => {
        // 确保 sliderRef.current 存在（不是 null）
        if (sliderRef.current) {
            const slider = sliderRef.current;
            const items = slider.children; // 获取 slider 下所有的子元素 (.box)

            // 原理：appendChild 会把已存在的元素移动到末尾
            if (items.length > 0) {
                slider.appendChild(items[0]);
            }
        }
    }

    // 上一张的逻辑
    const handlePrev = () => {
        if (sliderRef.current) {
            const slider = sliderRef.current;
            const items = slider.children;

            // 原理：prepend 会把已存在的元素移动到开头
            if (items.length > 0) {
                slider.prepend(items[items.length - 1]);
            }
        }
    }


    return (
        <div className='container'>
            <div className='slider' ref={sliderRef}>

                <div className='box'>
                    <img src={Banana} alt="banana" className='preview'/>
                    <div className='clipBox'>
                        <video src={banana} muted loop playsInline autoPlay />
                    </div>
                </div>

                <div className='box'>
                    <img src={Hello} alt="hello" className='preview'/>
                    <div className='clipBox'>
                        <video src="/video/hello.mp4" muted loop playsInline />
                    </div>
                </div>

                <div className='box'>
                    <img src={Baby} alt="baby" className='preview'/>
                    <div className='clipBox'>
                        <video src="/video/baby.mp4" muted loop playsInline />
                    </div>
                </div>

                <div className='box'>
                    <img src={Bread} alt="bread" className='preview'/>
                    <div className='clipBox'>
                        <video src="/video/bread.mp4" muted loop playsInline />
                    </div>
                </div>

                <div className='box'>
                    <img src={Halloween} alt="Halloween" className='preview'/>
                    <div className='clipBox'>
                        <video src="/video/halloween.mp4" muted loop playsInline />
                    </div>
                </div>

            </div>

            <div className='buttons'>
                <span className='prev' onClick={handlePrev}>&lt;</span>
                <span className='next' onClick={handleNext}>&gt;</span>
            </div>
        </div>
    )
}