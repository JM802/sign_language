import {useLocation} from "react-router-dom";
import {useEffect} from "react";

export const BackgroundController = () => {
    const location = useLocation();

    useEffect(() => {
        document.body.classList.remove('bg-gradient', 'bg-home');

        switch (location.pathname) {
            case '/learning':
                document.body.classList.add('bg-gradient');
                break;
            case '/about':
                document.body.classList.add('bg-gradient');
                break;
            case '/':
                document.body.classList.add('bg-home');
                break;
        }
    }, [location.pathname]);

    return null;
};
