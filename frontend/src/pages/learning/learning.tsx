import Banana from '../../assets/image/banana.png'
import Hello from '../../assets/image/hello.png'
import Baby from '../../assets/image/baby.png'
import Bread from '../../assets/image/bread.png'
import Halloween from '../../assets/image/halloween.png'
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
                <img src={Banana} alt="banana" className='box' />
                <img src={Hello} alt="hello" className='box' />
                <img src={Baby} alt="baby" className='box' />
                <img src={Bread} alt="bread" className='box' />
                <img src={Halloween} alt="Halloween" className='box' />
            </div>
            <div className='buttons'>
                <span className='prev' onClick={handlePrev}>&lt;</span>
                <span className='prev' onClick={handleNext}>&gt;</span>
            </div>
        </div>
    )
}