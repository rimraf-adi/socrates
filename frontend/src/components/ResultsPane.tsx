"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github-dark.css";

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
}: ResultsPaneProps) {
    if (error) {
        return (
            <div className="glass-panel p-6 text-center border-red-500/30">
                <p className="text-red-500 text-sm">{error}</p>
            </div>
        );
    }

    if (isLoading) {
        return (
            <div className="glass-panel p-12 text-center">
                <div className="w-6 h-6 border-2 border-[var(--text-muted)] border-t-[var(--accent)] rounded-full mx-auto mb-4 animate-spin" />
                <p className="text-[var(--text-muted)] text-sm">Thinking...</p>
            </div>
        );
    }

    if (!answer && sources.length === 0) {
        return null;
    }

    return (
        <div className="space-y-4">
            {/* Answer */}
            <div className="glass-panel p-6">
                <article className="prose prose-sm max-w-none 
                    prose-headings:text-[var(--text-primary)] prose-headings:font-medium
                    prose-p:text-[var(--text-secondary)] prose-p:leading-relaxed
                    prose-a:text-[var(--accent)] prose-a:no-underline hover:prose-a:underline
                    prose-strong:text-[var(--text-primary)]
                    prose-code:text-[var(--accent)] prose-code:bg-[var(--accent-muted)] prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-code:text-sm prose-code:before:content-none prose-code:after:content-none
                    prose-pre:bg-[#0d1117] prose-pre:border prose-pre:border-[var(--border)] prose-pre:rounded-lg prose-pre:p-4
                    prose-ul:text-[var(--text-secondary)] prose-ol:text-[var(--text-secondary)]
                    prose-blockquote:border-l-2 prose-blockquote:border-[var(--accent)] prose-blockquote:pl-4 prose-blockquote:text-[var(--text-muted)]
                ">
                    <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
                        {answer}
                    </ReactMarkdown>
                </article>
            </div>

            {/* Sources */}
            {sources.length > 0 && (
                <div className="glass-panel p-4">
                    <p className="text-xs text-[var(--text-muted)] mb-3">Sources</p>
                    <div className="flex flex-wrap gap-2">
                        {sources.slice(0, 10).map((source, i) => (
                            <a
                                key={i}
                                href={source.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-[var(--bg-muted)] hover:bg-[var(--accent-muted)] text-xs text-[var(--text-secondary)] hover:text-[var(--accent)] transition-colors"
                            >
                                <span className="font-medium">{i + 1}</span>
                                <span className="truncate max-w-[120px]">{new URL(source.url).hostname.replace('www.', '')}</span>
                            </a>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
