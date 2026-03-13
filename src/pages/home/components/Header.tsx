import React from 'react';
import { NavLink } from 'react-router-dom';
import { useBlinkNavigate } from '../../../shared/components/blinktransition.tsx';

const Header: React.FC = () => {
    const blinkNavigate = useBlinkNavigate();

    const navItems = [
        { name: 'Home', path: '/' },
        { name: 'About', path: '/about' },
        { name: 'Translate', path: '/translate' },
        { name: 'Learning', path: '/learning' },
        { name: 'Contacts', path: '/contacts' },
    ];

    const handleClick = (
        e: React.MouseEvent<HTMLAnchorElement>,
        to: string
    ) => {
        e.preventDefault();        // 阻止默认跳转
        blinkNavigate(to);         // 使用动画跳转
    };

    return (
        <header>
            <nav>
                {navItems.map((item) => (
                    <NavLink
                        key={item.name}
                        to={item.path}
                        onClick={(e) => handleClick(e, item.path)}
                        className={({ isActive }) =>
                            `nav-items ${isActive ? 'active' : ''}`
                        }
                    >
                        {item.name}
                    </NavLink>
                ))}
            </nav>
        </header>
    );
};

export default Header;