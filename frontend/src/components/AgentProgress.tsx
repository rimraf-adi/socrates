"use client";

interface ProgressEvent {
    type: string;
    node?: string;
    status?: string;
    message?: string;
    iteration?: number;
}

interface AgentProgressProps {
    progress: ProgressEvent;
    subQuestions: string[];
}

const nodeIcons: Record<string, string> = {
    plan: "📋",
    search: "🔍",
    analyze: "📊",
    synthesize: "✍️",
};

const nodeLabels: Record<string, string> = {
    plan: "Planning Research",
    search: "Searching",
    analyze: "Analyzing",
    synthesize: "Synthesizing",
};

export default function AgentProgress({
    progress,
    subQuestions,
}: AgentProgressProps) {
    const node = progress.node || "";

    return (
        <div className="bg-indigo-900/20 border border-indigo-700 rounded-xl p-4 mb-6">
            {/* Current Stage */}
            <div className="flex items-center gap-2 mb-3">
                <span className="text-2xl">{nodeIcons[node] || "⏳"}</span>
                <span className="text-indigo-400 font-semibold">
                    {nodeLabels[node] || node}
                </span>
                {progress.iteration && (
                    <span className="text-gray-500 text-sm ml-auto">
                        Iteration {progress.iteration}
                    </span>
                )}
            </div>

            {/* Progress Bar */}
            <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden mb-3">
                <div className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 animate-pulse" />
            </div>

            {/* Message */}
            {progress.message && (
                <p className="text-gray-300 text-sm mb-3">{progress.message}</p>
            )}

            {/* Sub-questions */}
            {subQuestions.length > 0 && (
                <div className="mt-3 pt-3 border-t border-gray-700">
                    <p className="text-gray-500 text-xs uppercase tracking-wide mb-2">
                        Research Areas
                    </p>
                    <div className="flex flex-wrap gap-2">
                        {subQuestions.map((q, i) => (
                            <span
                                key={i}
                                className="px-2 py-1 bg-gray-800 rounded text-sm text-gray-300"
                            >
                                {i + 1}. {q.substring(0, 40)}...
                            </span>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
