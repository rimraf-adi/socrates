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
    plan: "Planning",
    search: "Searching",
    analyze: "Analyzing",
    synthesize: "Synthesizing",
    evaluate: "Evaluating",
};

export default function AgentProgress({
    progress,
    subQuestions,
}: AgentProgressProps) {
    const node = progress.node || "";

    return (
        <div className="glass-panel p-5 mb-6">
            {/* Status */}
            <div className="flex items-center gap-3 mb-4">
                <div className="w-2 h-2 bg-[var(--accent)] rounded-full animate-pulse-subtle" />
                <span className="text-sm font-medium">
                    {nodeLabels[node] || node}
                </span>
                {progress.iteration && (
                    <span className="text-xs text-[var(--text-muted)] ml-auto">
                        #{progress.iteration}
                    </span>
                )}
            </div>

            {/* Progress Bar */}
            <div className="h-1 bg-[var(--bg-muted)] rounded-full overflow-hidden mb-3">
                <div className="h-full bg-[var(--accent)] animate-pulse-subtle" style={{ width: '100%' }} />
            </div>

            {/* Message */}
            {progress.message && (
                <p className="text-sm text-[var(--text-muted)]">{progress.message}</p>
            )}

            {/* Sub-questions */}
            {subQuestions.length > 0 && (
                <div className="mt-4 pt-4 border-t border-[var(--border)]">
                    <p className="text-xs text-[var(--text-muted)] mb-2">Exploring:</p>
                    <div className="flex flex-wrap gap-2">
                        {subQuestions.slice(0, 4).map((q, i) => (
                            <span key={i} className="px-2 py-1 bg-[var(--bg-muted)] rounded text-xs text-[var(--text-secondary)]">
                                {q.length > 30 ? q.slice(0, 30) + "..." : q}
                            </span>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
