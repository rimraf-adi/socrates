"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface SearchResult {
    title: string;
    url: string;
    snippet: string;
}

interface ResultsPaneProps {
    answer: string;
    sources: SearchResult[];
    isLoading: boolean;
    error: string | null;
    isDeepResearch?: boolean;
}

export default function ResultsPane({
    answer,
    sources,
    isLoading,
    error,
    isDeepResearch,
}: ResultsPaneProps) {
    if (error) {
        return (
            <div className="glass-panel border-[var(--error)]/30 p-8 text-center">
                <div className="text-2xl mb-3 opacity-50">!</div>
                <h3 className="text-[var(--error)] font-medium mb-2 text-sm">error</h3>
                <p className="text-[var(--text-muted)] text-sm">{error}</p>
            </div>
        );
    }

    if (isLoading) {
        return (
            <div className="glass-panel p-12 text-center">
                <div className="w-8 h-8 border-2 border-[var(--accent)]/30 border-t-[var(--accent)] rounded-full mx-auto mb-4 animate-spin" />
                <p className="text-[var(--text-muted)] text-sm">
                    {isDeepResearch
                        ? "conducting deep research..."
                        : "searching and synthesizing..."}
                </p>
            </div>
        );
    }

    if (!answer && sources.length === 0) {
        return (
            <div className="glass-panel p-12 text-center">
                <div className="text-2xl mb-3 opacity-30">?</div>
                <p className="text-[var(--text-muted)] text-sm">enter a query to start searching</p>
                {isDeepResearch && (
                    <p className="text-[var(--text-muted)] text-xs mt-2 opacity-70">
                        deep research mode will explore multiple angles
                    </p>
                )}
            </div>
        );
    }

    return (
        <div className="space-y-5">
            {/* Answer */}
            <div className="glass-panel p-6">
                <h2 className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider mb-4">
                    {isDeepResearch ? "research report" : "answer"}
                </h2>
                <div className="prose prose-sm max-w-none text-[var(--text-secondary)] prose-headings:text-[var(--text-primary)] prose-headings:font-medium prose-p:leading-relaxed prose-a:text-[var(--accent)] prose-strong:text-[var(--text-primary)] prose-code:text-[var(--text-primary)] prose-code:bg-[var(--bg-muted)] prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-code:text-xs">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{answer}</ReactMarkdown>
                </div>
            </div>

            {/* Sources */}
            {sources.length > 0 && (
                <div className="glass-panel p-6">
                    <h2 className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider mb-4">
                        sources ({sources.length})
                    </h2>
                    <div className="space-y-2">
                        {sources.slice(0, 15).map((source, i) => (
                            <a
                                key={i}
                                href={source.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="flex items-center gap-3 p-3 rounded-lg hover:bg-[var(--bg-muted)] transition-colors group"
                            >
                                <span className="flex-shrink-0 w-6 h-6 bg-[var(--accent)] rounded text-white text-xs font-medium flex items-center justify-center">
                                    {i + 1}
                                </span>
                                <div className="min-w-0 flex-1">
                                    <p className="text-[var(--text-primary)] text-sm truncate group-hover:text-[var(--accent)] transition-colors">
                                        {source.title}
                                    </p>
                                    <p className="text-[var(--text-muted)] text-xs truncate">
                                        {new URL(source.url).hostname}
                                    </p>
                                </div>
                            </a>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
