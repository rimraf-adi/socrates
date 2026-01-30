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
            <div className="bg-red-900/20 border border-red-700 rounded-xl p-8 text-center">
                <div className="text-4xl mb-4">⚠️</div>
                <h3 className="text-red-400 font-semibold mb-2">Error</h3>
                <p className="text-gray-400">{error}</p>
            </div>
        );
    }

    if (isLoading) {
        return (
            <div className="bg-gray-800/30 border border-gray-700 rounded-xl p-12 text-center">
                <div className="w-12 h-12 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full mx-auto mb-4 animate-pulse" />
                <p className="text-gray-400">
                    {isDeepResearch
                        ? "Conducting deep research..."
                        : "Searching and synthesizing..."}
                </p>
            </div>
        );
    }

    if (!answer && sources.length === 0) {
        return (
            <div className="bg-gray-800/30 border border-gray-700 rounded-xl p-12 text-center">
                <div className="text-4xl mb-4">🔍</div>
                <p className="text-gray-400">Enter a query to start searching</p>
                {isDeepResearch && (
                    <p className="text-gray-500 text-sm mt-2">
                        Deep Research mode will explore multiple angles of your question
                    </p>
                )}
            </div>
        );
    }

    return (
        <div className="space-y-6">
            {/* Answer */}
            <div className="bg-gray-800/30 border border-gray-700 rounded-xl p-6">
                <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    {isDeepResearch ? "📋 Research Report" : "💡 Answer"}
                </h2>
                <div className="prose prose-invert max-w-none prose-headings:text-white prose-p:text-gray-300 prose-li:text-gray-300 prose-strong:text-white prose-a:text-indigo-400">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{answer}</ReactMarkdown>
                </div>
            </div>

            {/* Sources */}
            {sources.length > 0 && (
                <div className="bg-gray-800/30 border border-gray-700 rounded-xl p-6">
                    <h2 className="text-lg font-semibold text-white mb-4">
                        📚 Sources ({sources.length})
                    </h2>
                    <div className="space-y-2">
                        {sources.slice(0, 15).map((source, i) => (
                            <a
                                key={i}
                                href={source.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="flex items-center gap-3 p-3 bg-gray-800/50 rounded-lg hover:bg-gray-700/50 transition-colors group"
                            >
                                <span className="flex-shrink-0 w-7 h-7 bg-gradient-to-r from-indigo-500 to-purple-500 rounded text-white text-sm font-semibold flex items-center justify-center">
                                    {i + 1}
                                </span>
                                <div className="min-w-0 flex-1">
                                    <p className="text-white font-medium truncate group-hover:text-indigo-400 transition-colors">
                                        {source.title}
                                    </p>
                                    <p className="text-gray-500 text-sm truncate">
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
