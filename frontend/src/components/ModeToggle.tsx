"use client";

interface ModeToggleProps {
    mode: "simple" | "deep";
    onModeChange: (mode: "simple" | "deep") => void;
}

export default function ModeToggle({ mode, onModeChange }: ModeToggleProps) {
    return (
        <div className="flex justify-center">
            <div className="inline-flex glass-panel p-1">
                <button
                    onClick={() => onModeChange("simple")}
                    className={`px-4 py-2 rounded-lg text-xs font-medium transition-all ${mode === "simple"
                        ? "bg-[var(--accent)] text-white"
                        : "text-[var(--text-muted)] hover:text-[var(--text-primary)]"
                        }`}
                >
                    simple search
                </button>
                <button
                    onClick={() => onModeChange("deep")}
                    className={`px-4 py-2 rounded-lg text-xs font-medium transition-all ${mode === "deep"
                        ? "bg-[var(--accent)] text-white"
                        : "text-[var(--text-muted)] hover:text-[var(--text-primary)]"
                        }`}
                >
                    deep research
                </button>
            </div>
        </div>
    );
}
