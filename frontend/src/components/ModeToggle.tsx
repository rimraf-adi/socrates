"use client";

interface ModeToggleProps {
    mode: "simple" | "deep";
    onModeChange: (mode: "simple" | "deep") => void;
}

export default function ModeToggle({ mode, onModeChange }: ModeToggleProps) {
    return (
        <div className="flex justify-center">
            <div className="inline-flex bg-gray-800/50 border border-gray-700 rounded-lg p-1">
                <button
                    onClick={() => onModeChange("simple")}
                    className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${mode === "simple"
                            ? "bg-gradient-to-r from-indigo-500 to-purple-500 text-white shadow-lg"
                            : "text-gray-400 hover:text-white"
                        }`}
                >
                    ⚡ Simple Search
                </button>
                <button
                    onClick={() => onModeChange("deep")}
                    className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${mode === "deep"
                            ? "bg-gradient-to-r from-indigo-500 to-purple-500 text-white shadow-lg"
                            : "text-gray-400 hover:text-white"
                        }`}
                >
                    🔬 Deep Research
                </button>
            </div>
        </div>
    );
}
