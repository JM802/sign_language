import Header from './pages/home/components/Header';
import Home from './pages/home/home.tsx';
import About from './pages/about/about.tsx';
import Translate from './pages/translate/translate.tsx'
import Learning from './pages/learning/learning.tsx';
import Contact from './pages/contact/contact.tsx';``

import {BrowserRouter, Route, Routes} from "react-router-dom";
import {BlinkTransitionProvider} from "./shared/components/blinktransition.tsx";
import {BackgroundController} from "./shared/components/BackgroundController.tsx";


function App() {
    return (
        <BrowserRouter>
            <BlinkTransitionProvider>
                <BackgroundController />
                <div className="app-wrapper">
                    <Header />

                     <Routes>
                            <Route path="/" element={<Home />} />
                            <Route path="/about" element={<About />} />
                            <Route path="/translate" element={<Translate />} />
                            <Route path="/learning" element={<Learning />} />
                            <Route path="/contacts" element={<Contact />} />

                        </Routes>
             </div>
            </BlinkTransitionProvider>
        </BrowserRouter>
    );
}

export default App;