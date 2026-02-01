"use client";

interface ResearchSettingsProps {
    depth: "quick" | "standard" | "deep" | "exhaustive";
    maxIterations: number | null;
    onDepthChange: (depth: "quick" | "standard" | "deep" | "exhaustive") => void;
    onMaxIterationsChange: (iterations: number | null) => void;
}

const DEPTH_INFO = {
    quick: { label: "Quick", iterations: 5, description: "Fast overview" },
    standard: { label: "Standard", iterations: 10, description: "Balanced research" },
    deep: { label: "Deep", iterations: 15, description: "Thorough analysis" },
    exhaustive: { label: "Exhaustive", iterations: 20, description: "Maximum depth" },
};

export default function ResearchSettings({
    depth,
    maxIterations,
    onDepthChange,
    onMaxIterationsChange,
}: ResearchSettingsProps) {
    const currentIterations = maxIterations ?? DEPTH_INFO[depth].iterations;

    return (
        <div className="glass-panel p-4 space-y-4">
            {/* Depth Selector */}
            <div className="space-y-2">
                <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider">
                    Research Depth
                </label>
                <div className="flex gap-2">
                    {(Object.keys(DEPTH_INFO) as Array<keyof typeof DEPTH_INFO>).map((key) => (
                        <button
                            key={key}
                            onClick={() => {
                                onDepthChange(key);
                                onMaxIterationsChange(null); // Reset custom iterations when depth changes
                            }}
                            className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium transition-all ${depth === key
                                    ? "bg-[var(--accent)] text-white"
                                    : "bg-[var(--bg-muted)] text-[var(--text-secondary)] hover:bg-[var(--bg-elevated)]"
                                }`}
                            title={DEPTH_INFO[key].description}
                        >
                            {DEPTH_INFO[key].label}
                        </button>
                    ))}
                </div>
            </div>

            {/* Iterations Slider */}
            <div className="space-y-2">
                <div className="flex justify-between items-center">
                    <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider">
                        Max Iterations
                    </label>
                    <span className="text-sm font-medium text-[var(--accent)]">
                        {currentIterations}
                    </span>
                </div>
                <input
                    type="range"
                    min="3"
                    max="25"
                    value={currentIterations}
                    onChange={(e) => onMaxIterationsChange(parseInt(e.target.value))}
                    className="w-full accent-[var(--accent)] cursor-pointer"
                />
                <div className="flex justify-between text-xs text-[var(--text-muted)]">
                    <span>Faster</span>
                    <span>More thorough</span>
                </div>
            </div>
        </div>
    );
}
