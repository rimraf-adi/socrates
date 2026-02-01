"use client";

interface ResearchSettingsProps {
    depth: "quick" | "standard" | "deep" | "exhaustive";
    maxIterations: number | null;
    onDepthChange: (depth: "quick" | "standard" | "deep" | "exhaustive") => void;
    onMaxIterationsChange: (iterations: number | null) => void;
}

const DEPTH_INFO = {
    quick: { label: "Quick", iterations: 5, icon: "⚡" },
    standard: { label: "Standard", iterations: 10, icon: "🔍" },
    deep: { label: "Deep", iterations: 15, icon: "🔬" },
    exhaustive: { label: "Exhaustive", iterations: 20, icon: "🧠" },
};

export default function ResearchSettings({
    depth,
    maxIterations,
    onDepthChange,
    onMaxIterationsChange,
}: ResearchSettingsProps) {
    const currentIterations = maxIterations ?? DEPTH_INFO[depth].iterations;

    return (
        <div className="glass-panel p-5 space-y-5">
            {/* Depth Selector */}
            <div className="space-y-3">
                <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider font-medium">
                    Research Depth
                </label>
                <div className="grid grid-cols-4 gap-2">
                    {(Object.keys(DEPTH_INFO) as Array<keyof typeof DEPTH_INFO>).map((key) => (
                        <button
                            key={key}
                            onClick={() => {
                                onDepthChange(key);
                                onMaxIterationsChange(null);
                            }}
                            className={`px-3 py-3 rounded-xl text-sm font-medium transition-all duration-200 flex flex-col items-center gap-1 ${depth === key
                                    ? "bg-gradient-to-br from-purple-500/20 to-cyan-500/20 text-[var(--text-primary)] border border-purple-500/30 shadow-lg shadow-purple-500/10"
                                    : "bg-[var(--bg-muted)] text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)] border border-transparent"
                                }`}
                        >
                            <span className="text-lg">{DEPTH_INFO[key].icon}</span>
                            <span>{DEPTH_INFO[key].label}</span>
                        </button>
                    ))}
                </div>
            </div>

            {/* Iterations Slider */}
            <div className="space-y-3">
                <div className="flex justify-between items-center">
                    <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider font-medium">
                        Max Iterations
                    </label>
                    <span className="text-sm font-bold gradient-text">
                        {currentIterations}
                    </span>
                </div>
                <div className="relative">
                    <input
                        type="range"
                        min="3"
                        max="25"
                        value={currentIterations}
                        onChange={(e) => onMaxIterationsChange(parseInt(e.target.value))}
                        className="w-full h-2 bg-[var(--bg-subtle)] rounded-full appearance-none cursor-pointer 
              [&::-webkit-slider-thumb]:appearance-none 
              [&::-webkit-slider-thumb]:w-5 
              [&::-webkit-slider-thumb]:h-5 
              [&::-webkit-slider-thumb]:rounded-full 
              [&::-webkit-slider-thumb]:bg-gradient-to-r 
              [&::-webkit-slider-thumb]:from-purple-500 
              [&::-webkit-slider-thumb]:to-cyan-500 
              [&::-webkit-slider-thumb]:shadow-lg 
              [&::-webkit-slider-thumb]:shadow-purple-500/30
              [&::-webkit-slider-thumb]:cursor-pointer
              [&::-webkit-slider-thumb]:transition-transform
              [&::-webkit-slider-thumb]:hover:scale-110"
                    />
                </div>
                <div className="flex justify-between text-xs text-[var(--text-muted)]">
                    <span>⚡ Faster</span>
                    <span>More thorough 🔬</span>
                </div>
            </div>
        </div>
    );
}
