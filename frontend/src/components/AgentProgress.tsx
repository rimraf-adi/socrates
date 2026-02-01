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

const nodeConfig: Record<string, { label: string; icon: string }> = {
    plan: { label: "Planning research strategy", icon: "🧠" },
    search: { label: "Searching the web", icon: "🔍" },
    analyze: { label: "Analyzing results", icon: "📊" },
    synthesize: { label: "Synthesizing answer", icon: "✨" },
    evaluate: { label: "Evaluating quality", icon: "⚖️" },
};

export default function AgentProgress({
    progress,
    subQuestions,
}: AgentProgressProps) {
    const node = progress.node || "";
    const config = nodeConfig[node] || { label: node, icon: "⚡" };

    return (
        <div className="glass-panel p-6 mb-6 border border-purple-500/20 bg-gradient-to-r from-purple-500/5 to-cyan-500/5">
            {/* Header */}
            <div className="flex items-center gap-3 mb-5">
                <div className="relative">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-cyan-500 flex items-center justify-center text-lg">
                        {config.icon}
                    </div>
                    <div className="absolute -bottom-0.5 -right-0.5 w-3 h-3 bg-green-400 rounded-full border-2 border-[var(--bg-elevated)] animate-pulse" />
                </div>
                <div className="flex-1">
                    <p className="text-[var(--text-primary)] font-semibold">
                        {config.label}
                    </p>
                    <p className="text-[var(--text-muted)] text-xs">
                        {progress.message || "Working..."}
                    </p>
                </div>
                {progress.iteration && (
                    <div className="px-3 py-1.5 rounded-lg bg-[var(--bg-muted)] border border-[var(--border)]">
                        <span className="text-xs text-[var(--text-muted)]">Iteration</span>
                        <span className="ml-2 text-sm font-bold gradient-text">{progress.iteration}</span>
                    </div>
                )}
            </div>

            {/* Progress Bar */}
            <div className="h-1.5 bg-[var(--bg-muted)] rounded-full overflow-hidden mb-5">
                <div
                    className="h-full bg-gradient-to-r from-purple-500 to-cyan-500 rounded-full animate-shimmer"
                    style={{ width: '100%' }}
                />
            </div>

            {/* Sub-questions */}
            {subQuestions.length > 0 && (
                <div className="pt-4 border-t border-[var(--border)]">
                    <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-3 font-medium">
                        Research areas explored
                    </p>
                    <div className="flex flex-wrap gap-2">
                        {subQuestions.map((q, i) => (
                            <span
                                key={i}
                                className="px-3 py-1.5 bg-gradient-to-r from-purple-500/10 to-cyan-500/10 border border-purple-500/20 rounded-lg text-xs text-[var(--text-secondary)] font-medium"
                            >
                                <span className="gradient-text mr-1">{i + 1}.</span>
                                {q.length > 35 ? q.substring(0, 35) + "..." : q}
                            </span>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
