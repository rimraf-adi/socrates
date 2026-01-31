"use client";

import { createContext, useContext, useState, useEffect, ReactNode } from "react";

interface ThemeContextType {
    theme: "light" | "dark";
    toggleTheme: () => void;
    glassOpacity: number;
    setGlassOpacity: (opacity: number) => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export function useTheme() {
    const context = useContext(ThemeContext);
    if (!context) {
        throw new Error("useTheme must be used within a ThemeProvider");
    }
    return context;
}

interface ThemeProviderProps {
    children: ReactNode;
}

export function ThemeProvider({ children }: ThemeProviderProps) {
    const [theme, setTheme] = useState<"light" | "dark">("dark");
    const [glassOpacity, setGlassOpacity] = useState(0.6);
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
        const savedTheme = localStorage.getItem("theme") as "light" | "dark" | null;
        const savedOpacity = localStorage.getItem("glassOpacity");

        if (savedTheme) {
            setTheme(savedTheme);
        }
        if (savedOpacity) {
            setGlassOpacity(parseFloat(savedOpacity));
        }
    }, []);

    useEffect(() => {
        if (!mounted) return;

        const root = document.documentElement;

        if (theme === "dark") {
            root.classList.add("dark");
        } else {
            root.classList.remove("dark");
        }

        root.style.setProperty("--glass-opacity", glassOpacity.toString());

        localStorage.setItem("theme", theme);
        localStorage.setItem("glassOpacity", glassOpacity.toString());
    }, [theme, glassOpacity, mounted]);

    const toggleTheme = () => {
        setTheme(prev => prev === "dark" ? "light" : "dark");
    };

    return (
        <ThemeContext.Provider value={{ theme, toggleTheme, glassOpacity, setGlassOpacity }}>
            {children}
        </ThemeContext.Provider>
    );
}
