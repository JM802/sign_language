import HeroSection from './components/HeroSection.tsx';  // 可选抽离
import '../../styles/styles.css'

export default function Home() {
    return (
        <div className="home-wrapper">
            <HeroSection />
        </div>
    );
}