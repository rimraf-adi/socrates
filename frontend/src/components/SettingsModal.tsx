"use client";

import { useState, useEffect } from "react";

type AccentColor = "neutral" | "purple" | "blue" | "green" | "orange" | "pink" | "cyan";
type Provider = "lmstudio" | "gemini" | "hybrid";
type Depth = "quick" | "standard" | "deep" | "exhaustive";

interface SettingsModalProps {
    isOpen: boolean;
    onClose: () => void;
    researchSettings: {
        provider: Provider;
        depth: Depth;
        maxIterations: number;
    };
    onResearchSettingsChange: (settings: { provider: Provider; depth: Depth; maxIterations: number }) => void;
}

const ACCENT_COLORS: { id: AccentColor; label: string; lightColor: string; darkColor: string }[] = [
    { id: "neutral", label: "Neutral", lightColor: "#171717", darkColor: "#fafafa" },
    { id: "purple", label: "Purple", lightColor: "#8b5cf6", darkColor: "#a78bfa" },
    { id: "blue", label: "Blue", lightColor: "#3b82f6", darkColor: "#60a5fa" },
    { id: "green", label: "Green", lightColor: "#22c55e", darkColor: "#4ade80" },
    { id: "orange", label: "Orange", lightColor: "#f97316", darkColor: "#fb923c" },
    { id: "pink", label: "Pink", lightColor: "#ec4899", darkColor: "#f472b6" },
    { id: "cyan", label: "Cyan", lightColor: "#06b6d4", darkColor: "#22d3ee" },
];

const PROVIDERS: { id: Provider; label: string; desc: string }[] = [
    { id: "lmstudio", label: "Local", desc: "LMStudio only" },
    { id: "gemini", label: "Gemini", desc: "Google AI only" },
    { id: "hybrid", label: "Hybrid", desc: "Local + Gemini" },
];

const DEPTHS: { id: Depth; label: string; iterations: number }[] = [
    { id: "quick", label: "Quick", iterations: 5 },
    { id: "standard", label: "Standard", iterations: 10 },
    { id: "deep", label: "Deep", iterations: 15 },
    { id: "exhaustive", label: "Exhaustive", iterations: 20 },
];

export default function SettingsModal({ isOpen, onClose, researchSettings, onResearchSettingsChange }: SettingsModalProps) {
    const [accent, setAccent] = useState<AccentColor>("neutral");
    const [isDark, setIsDark] = useState(true);

    useEffect(() => {
        const savedAccent = localStorage.getItem("accent") as AccentColor | null;
        const savedDark = localStorage.getItem("dark");
        if (savedAccent) setAccent(savedAccent);
        // Default to dark if not set
        setIsDark(savedDark === null ? true : savedDark === "true");
    }, [isOpen]);

    const handleAccentChange = (newAccent: AccentColor) => {
        setAccent(newAccent);
        localStorage.setItem("accent", newAccent);

        // Remove all accent classes from html element
        const html = document.documentElement;
        ACCENT_COLORS.forEach(c => html.classList.remove(`accent-${c.id}`));

        // Add new accent class if not neutral
        if (newAccent !== "neutral") {
            html.classList.add(`accent-${newAccent}`);
        }
    };

    const handleDarkToggle = () => {
        const newDark = !isDark;
        setIsDark(newDark);
        localStorage.setItem("dark", String(newDark));

        const html = document.documentElement;
        if (newDark) {
            html.classList.add("dark");
        } else {
            html.classList.remove("dark");
        }
    };

    if (!isOpen) return null;

    return (
        <>
            <div className="fixed inset-0 bg-black/50 z-50" onClick={onClose} />
            <div className="fixed inset-0 z-50 flex items-center justify-center p-4 overflow-y-auto">
                <div className="w-full max-w-md glass-panel-strong p-6 my-8" onClick={e => e.stopPropagation()}>
                    {/* Header */}
                    <div className="flex items-center justify-between mb-6">
                        <h2 className="text-lg font-medium text-[var(--text-primary)]">Settings</h2>
                        <button onClick={onClose} className="p-1 hover:bg-[var(--bg-muted)] rounded-lg">
                            <svg className="w-5 h-5 text-[var(--text-muted)]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                            </svg>
                        </button>
                    </div>

                    {/* Appearance Section */}
                    <div className="mb-6">
                        <h3 className="text-xs text-[var(--text-muted)] uppercase tracking-wide mb-3">Appearance</h3>

                        {/* Theme Toggle */}
                        <button
                            onClick={handleDarkToggle}
                            className="w-full flex items-center justify-between p-3 rounded-lg border border-[var(--border)] hover:bg-[var(--bg-muted)] mb-3"
                        >
                            <span className="text-sm text-[var(--text-primary)]">{isDark ? "Dark Mode" : "Light Mode"}</span>
                            <div className={`w-10 h-6 rounded-full p-1 transition-colors ${isDark ? 'bg-[var(--text-primary)]' : 'bg-[var(--border)]'}`}>
                                <div className={`w-4 h-4 rounded-full bg-[var(--bg-base)] transition-transform ${isDark ? 'translate-x-4' : ''}`} />
                            </div>
                        </button>

                        {/* Accent Colors */}
                        <div className="grid grid-cols-7 gap-2">
                            {ACCENT_COLORS.map((c) => (
                                <button
                                    key={c.id}
                                    onClick={() => handleAccentChange(c.id)}
                                    className={`aspect-square rounded-lg border-2 transition-all flex items-center justify-center ${accent === c.id ? 'border-[var(--text-primary)] scale-110' : 'border-transparent hover:border-[var(--border)]'
                                        }`}
                                    title={c.label}
                                >
                                    <div
                                        className="w-5 h-5 rounded-full"
                                        style={{ background: isDark ? c.darkColor : c.lightColor }}
                                    />
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Deep Research Section */}
                    <div className="pt-6 border-t border-[var(--border)]">
                        <h3 className="text-xs text-[var(--text-muted)] uppercase tracking-wide mb-3">Deep Research</h3>

                        {/* Provider */}
                        <div className="mb-4">
                            <label className="text-xs text-[var(--text-muted)] mb-2 block">Provider</label>
                            <div className="grid grid-cols-3 gap-2">
                                {PROVIDERS.map((p) => (
                                    <button
                                        key={p.id}
                                        onClick={() => onResearchSettingsChange({ ...researchSettings, provider: p.id })}
                                        className={`p-3 rounded-lg border text-center transition-colors ${researchSettings.provider === p.id
                                                ? 'border-[var(--text-primary)] bg-[var(--bg-muted)]'
                                                : 'border-[var(--border)] hover:bg-[var(--bg-muted)]'
                                            }`}
                                    >
                                        <div className="text-sm font-medium text-[var(--text-primary)]">{p.label}</div>
                                        <div className="text-xs text-[var(--text-muted)]">{p.desc}</div>
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Depth */}
                        <div className="mb-4">
                            <label className="text-xs text-[var(--text-muted)] mb-2 block">Research Depth</label>
                            <div className="grid grid-cols-4 gap-2">
                                {DEPTHS.map((d) => (
                                    <button
                                        key={d.id}
                                        onClick={() => onResearchSettingsChange({
                                            ...researchSettings,
                                            depth: d.id,
                                            maxIterations: d.iterations
                                        })}
                                        className={`p-2 rounded-lg border text-center transition-colors ${researchSettings.depth === d.id
                                                ? 'border-[var(--text-primary)] bg-[var(--bg-muted)]'
                                                : 'border-[var(--border)] hover:bg-[var(--bg-muted)]'
                                            }`}
                                    >
                                        <div className="text-xs font-medium text-[var(--text-primary)]">{d.label}</div>
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Max Iterations */}
                        <div>
                            <div className="flex justify-between mb-2">
                                <label className="text-xs text-[var(--text-muted)]">Max Iterations</label>
                                <span className="text-xs font-medium text-[var(--text-primary)]">{researchSettings.maxIterations}</span>
                            </div>
                            <input
                                type="range"
                                min="3"
                                max="25"
                                value={researchSettings.maxIterations}
                                onChange={(e) => onResearchSettingsChange({ ...researchSettings, maxIterations: parseInt(e.target.value) })}
                                className="w-full h-1.5 bg-[var(--bg-muted)] rounded-full appearance-none cursor-pointer
                  [&::-webkit-slider-thumb]:appearance-none
                  [&::-webkit-slider-thumb]:w-4
                  [&::-webkit-slider-thumb]:h-4
                  [&::-webkit-slider-thumb]:rounded-full
                  [&::-webkit-slider-thumb]:bg-[var(--text-primary)]
                  [&::-webkit-slider-thumb]:cursor-pointer"
                            />
                            <div className="flex justify-between text-xs text-[var(--text-muted)] mt-1">
                                <span>Faster</span>
                                <span>More thorough</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </>
    );
}
