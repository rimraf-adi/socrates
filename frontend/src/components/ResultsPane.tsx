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
    isDeepResearch,
}: ResultsPaneProps) {
    if (error) {
        return (
            <div className="glass-panel border border-red-500/30 p-8 text-center bg-gradient-to-br from-red-500/5 to-transparent">
                <div className="w-12 h-12 mx-auto mb-4 rounded-xl bg-red-500/10 flex items-center justify-center">
                    <svg className="w-6 h-6 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
                    </svg>
                </div>
                <h3 className="text-red-400 font-semibold mb-2">Something went wrong</h3>
                <p className="text-[var(--text-muted)] text-sm">{error}</p>
            </div>
        );
    }

    if (isLoading) {
        return (
            <div className="glass-panel p-12 text-center">
                <div className="relative w-12 h-12 mx-auto mb-5">
                    <div className="absolute inset-0 rounded-full bg-gradient-to-r from-purple-500 to-cyan-500 animate-spin" style={{ padding: '2px' }}>
                        <div className="w-full h-full rounded-full bg-[var(--bg-elevated)]" />
                    </div>
                    <div className="absolute inset-2 rounded-full bg-gradient-to-r from-purple-500 to-cyan-500 opacity-30 blur-sm" />
                </div>
                <p className="text-[var(--text-secondary)] font-medium">
                    {isDeepResearch
                        ? "Conducting deep research..."
                        : "Searching and synthesizing..."}
                </p>
                <p className="text-[var(--text-muted)] text-sm mt-2">
                    This may take a moment
                </p>
            </div>
        );
    }

    if (!answer && sources.length === 0) {
        return (
            <div className="glass-panel p-12 text-center">
                <div className="w-16 h-16 mx-auto mb-5 rounded-2xl bg-gradient-to-br from-purple-500/10 to-cyan-500/10 flex items-center justify-center">
                    <svg className="w-8 h-8 text-[var(--text-muted)]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456z" />
                    </svg>
                </div>
                <h3 className="text-[var(--text-primary)] font-semibold mb-2">Ready to research</h3>
                <p className="text-[var(--text-muted)] text-sm">
                    Enter a query above to start exploring
                </p>
                {isDeepResearch && (
                    <p className="text-[var(--text-muted)] text-xs mt-3 px-8">
                        Deep research will explore multiple angles and synthesize comprehensive answers
                    </p>
                )}
            </div>
        );
    }

    return (
        <div className="space-y-5">
            {/* Answer */}
            <div className="glass-panel-strong p-8">
                <div className="flex items-center gap-3 mb-6">
                    <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-purple-500 to-cyan-500 flex items-center justify-center">
                        <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
                        </svg>
                    </div>
                    <h2 className="text-sm font-semibold text-[var(--text-primary)]">
                        {isDeepResearch ? "Research Report" : "Answer"}
                    </h2>
                </div>
                <article className="prose prose-sm max-w-none 
                    prose-headings:text-[var(--text-primary)] prose-headings:font-semibold prose-headings:mt-6 prose-headings:mb-3
                    prose-h1:text-2xl prose-h1:border-b prose-h1:border-[var(--border)] prose-h1:pb-2
                    prose-h2:text-xl prose-h2:mt-8
                    prose-h3:text-lg
                    prose-p:text-[var(--text-secondary)] prose-p:leading-relaxed prose-p:my-3
                    prose-a:text-purple-400 prose-a:no-underline hover:prose-a:underline
                    prose-strong:text-[var(--text-primary)] prose-strong:font-semibold
                    prose-em:text-[var(--text-secondary)] prose-em:italic
                    prose-code:text-purple-400 prose-code:bg-purple-500/10 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded-md prose-code:text-sm prose-code:font-normal prose-code:before:content-none prose-code:after:content-none
                    prose-pre:bg-[#0d1117] prose-pre:border prose-pre:border-[var(--border)] prose-pre:rounded-xl prose-pre:p-4 prose-pre:overflow-x-auto prose-pre:my-4
                    prose-ul:my-3 prose-ul:space-y-1 prose-ul:list-disc prose-ul:pl-5
                    prose-ol:my-3 prose-ol:space-y-1 prose-ol:list-decimal prose-ol:pl-5
                    prose-li:text-[var(--text-secondary)] prose-li:my-0.5
                    prose-blockquote:border-l-4 prose-blockquote:border-purple-500 prose-blockquote:pl-4 prose-blockquote:italic prose-blockquote:text-[var(--text-muted)] prose-blockquote:my-4
                    prose-hr:border-[var(--border)] prose-hr:my-6
                    prose-table:border-collapse prose-table:w-full prose-table:my-4
                    prose-th:bg-[var(--bg-muted)] prose-th:p-3 prose-th:text-left prose-th:text-[var(--text-primary)] prose-th:font-semibold prose-th:border prose-th:border-[var(--border)]
                    prose-td:p-3 prose-td:border prose-td:border-[var(--border)] prose-td:text-[var(--text-secondary)]
                    prose-img:rounded-lg prose-img:my-4
                ">
                    <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        rehypePlugins={[rehypeHighlight]}
                    >
                        {answer}
                    </ReactMarkdown>
                </article>
            </div>

            {/* Sources */}
            {sources.length > 0 && (
                <div className="glass-panel p-6">
                    <div className="flex items-center gap-3 mb-5">
                        <div className="w-8 h-8 rounded-lg bg-[var(--bg-muted)] flex items-center justify-center">
                            <svg className="w-4 h-4 text-[var(--text-secondary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                                <path strokeLinecap="round" strokeLinejoin="round" d="M13.19 8.688a4.5 4.5 0 011.242 7.244l-4.5 4.5a4.5 4.5 0 01-6.364-6.364l1.757-1.757m13.35-.622l1.757-1.757a4.5 4.5 0 00-6.364-6.364l-4.5 4.5a4.5 4.5 0 001.242 7.244" />
                            </svg>
                        </div>
                        <h2 className="text-sm font-semibold text-[var(--text-primary)]">
                            Sources
                        </h2>
                        <span className="text-xs text-[var(--text-muted)] bg-[var(--bg-muted)] px-2 py-0.5 rounded-full">
                            {sources.length}
                        </span>
                    </div>
                    <div className="grid gap-2">
                        {sources.slice(0, 15).map((source, i) => (
                            <a
                                key={i}
                                href={source.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="flex items-center gap-3 p-3 rounded-xl hover:bg-gradient-to-r hover:from-purple-500/5 hover:to-cyan-500/5 border border-transparent hover:border-purple-500/20 transition-all duration-200 group"
                            >
                                <span className="flex-shrink-0 w-7 h-7 bg-gradient-to-br from-purple-500 to-cyan-500 rounded-lg text-white text-xs font-medium flex items-center justify-center shadow-sm">
                                    {i + 1}
                                </span>
                                <div className="min-w-0 flex-1">
                                    <p className="text-[var(--text-primary)] text-sm font-medium truncate group-hover:text-purple-400 transition-colors">
                                        {source.title}
                                    </p>
                                    <p className="text-[var(--text-muted)] text-xs truncate">
                                        {new URL(source.url).hostname}
                                    </p>
                                </div>
                                <svg className="w-4 h-4 text-[var(--text-muted)] opacity-0 group-hover:opacity-100 transition-opacity" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 6H5.25A2.25 2.25 0 003 8.25v10.5A2.25 2.25 0 005.25 21h10.5A2.25 2.25 0 0018 18.75V10.5m-10.5 6L21 3m0 0h-5.25M21 3v5.25" />
                                </svg>
                            </a>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
