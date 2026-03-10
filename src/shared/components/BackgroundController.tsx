import {useLocation} from "react-router-dom";
import {useEffect} from "react";

export const BackgroundController = () => {
    const location = useLocation();

    useEffect(() => {
        document.body.classList.remove('bg-coralPink', 'bg-gold', 'bg-brown', 'bg-purple', 'bg-home', 'page-long');
        switch (location.pathname) {
            case '/contacts':
                document.body.classList.add('bg-purple');
                break;
            case '/learning':
                document.body.classList.add('bg-coralPink');
                break;
            case '/translate':
                document.body.classList.add('bg-gold');
                break;
            case '/about':
                document.body.classList.add('bg-brown','page-long');
                break;
            case '/':
                document.body.classList.add('bg-home');
                break;
        }
    }, [location.pathname]);

    return null;
};
