"use client";

interface ProviderSelectorProps {
    provider: "lmstudio" | "gemini" | "hybrid";
    onProviderChange: (provider: "lmstudio" | "gemini" | "hybrid") => void;
}

export default function ProviderSelector({ provider, onProviderChange }: ProviderSelectorProps) {
    return (
        <div className="glass-panel p-1 flex gap-1">
            <button
                onClick={() => onProviderChange("lmstudio")}
                className={`px-3 py-2 rounded-lg text-sm font-medium transition-all ${provider === "lmstudio"
                        ? "bg-[var(--accent)] text-white"
                        : "text-[var(--text-secondary)] hover:bg-[var(--bg-muted)]"
                    }`}
                title="Local LLM via LMStudio"
            >
                <span className="flex items-center gap-1.5">
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 3v1.5M4.5 8.25H3m18 0h-1.5M4.5 12H3m18 0h-1.5m-15 3.75H3m18 0h-1.5M8.25 19.5V21M12 3v1.5m0 15V21m3.75-18v1.5m0 15V21m-9-1.5h10.5a2.25 2.25 0 002.25-2.25V6.75a2.25 2.25 0 00-2.25-2.25H6.75A2.25 2.25 0 004.5 6.75v10.5a2.25 2.25 0 002.25 2.25zm.75-12h9v9h-9v-9z" />
                    </svg>
                    Local
                </span>
            </button>
            <button
                onClick={() => onProviderChange("gemini")}
                className={`px-3 py-2 rounded-lg text-sm font-medium transition-all ${provider === "gemini"
                        ? "bg-[var(--accent)] text-white"
                        : "text-[var(--text-secondary)] hover:bg-[var(--bg-muted)]"
                    }`}
                title="Google Gemini API"
            >
                <span className="flex items-center gap-1.5">
                    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
                    </svg>
                    Gemini
                </span>
            </button>
            <button
                onClick={() => onProviderChange("hybrid")}
                className={`px-3 py-2 rounded-lg text-sm font-medium transition-all ${provider === "hybrid"
                        ? "bg-[var(--accent)] text-white"
                        : "text-[var(--text-secondary)] hover:bg-[var(--bg-muted)]"
                    }`}
                title="Local for light tasks, Gemini for synthesis"
            >
                <span className="flex items-center gap-1.5">
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 21L3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5" />
                    </svg>
                    Hybrid
                </span>
            </button>
        </div>
    );
}
