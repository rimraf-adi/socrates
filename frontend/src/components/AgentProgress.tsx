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

const nodeLabels: Record<string, string> = {
    plan: "planning",
    search: "searching",
    analyze: "analyzing",
    synthesize: "synthesizing",
};

export default function AgentProgress({
    progress,
    subQuestions,
}: AgentProgressProps) {
    const node = progress.node || "";

    return (
        <div className="glass-panel p-5 mb-6">
            {/* Current Stage */}
            <div className="flex items-center gap-3 mb-4">
                <div className="w-2 h-2 bg-[var(--accent)] rounded-full animate-pulse-subtle" />
                <span className="text-[var(--text-primary)] text-sm font-medium">
                    {nodeLabels[node] || node}
                </span>
                {progress.iteration && (
                    <span className="text-[var(--text-muted)] text-xs ml-auto">
                        iteration {progress.iteration}
                    </span>
                )}
            </div>

            {/* Progress Bar */}
            <div className="h-0.5 bg-[var(--border-default)] rounded-full overflow-hidden mb-4">
                <div className="h-full w-full bg-[var(--accent)] animate-pulse-subtle origin-left" />
            </div>

            {/* Message */}
            {progress.message && (
                <p className="text-[var(--text-secondary)] text-sm mb-4">{progress.message}</p>
            )}

            {/* Sub-questions */}
            {subQuestions.length > 0 && (
                <div className="pt-4 border-t border-[var(--border-default)]">
                    <p className="text-[var(--text-muted)] text-xs uppercase tracking-wider mb-3">
                        research areas
                    </p>
                    <div className="flex flex-wrap gap-2">
                        {subQuestions.map((q, i) => (
                            <span
                                key={i}
                                className="px-2.5 py-1 bg-[var(--bg-muted)] rounded text-xs text-[var(--text-secondary)]"
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
