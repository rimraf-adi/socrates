"use client";

import { useState, useEffect } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface HistoryItem {
    id: string;
    query: string;
    provider: string;
    depth: string;
    timestamp: string;
    source_count: number;
    dir_name: string;
}

interface SidebarProps {
    isOpen: boolean;
    onToggle: () => void;
    onSelectHistory: (item: HistoryItem) => void;
}

export default function Sidebar({ isOpen, onToggle, onSelectHistory }: SidebarProps) {
    const [history, setHistory] = useState<HistoryItem[]>([]);
    const [loading, setLoading] = useState(false);
    const [hoveredItem, setHoveredItem] = useState<string | null>(null);

    const fetchHistory = async () => {
        setLoading(true);
        try {
            const response = await fetch(`${API_URL}/api/history`);
            if (response.ok) {
                const data = await response.json();
                setHistory(data);
            }
        } catch (error) {
            console.error("Failed to fetch history:", error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        if (isOpen) {
            fetchHistory();
        }
    }, [isOpen]);

    const handleDelete = async (dirName: string, e: React.MouseEvent) => {
        e.stopPropagation();
        try {
            const response = await fetch(`${API_URL}/api/history/${encodeURIComponent(dirName)}`, {
                method: "DELETE",
            });
            if (response.ok) {
                setHistory((prev) => prev.filter((item) => item.dir_name !== dirName));
            }
        } catch (error) {
            console.error("Failed to delete:", error);
        }
    };

    const formatDate = (timestamp: string) => {
        const date = new Date(timestamp);
        const now = new Date();
        const diff = now.getTime() - date.getTime();
        const days = Math.floor(diff / (1000 * 60 * 60 * 24));

        if (days === 0) return "Today";
        if (days === 1) return "Yesterday";
        if (days < 7) return `${days}d ago`;
        return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
    };

    const truncateQuery = (query: string, maxLen: number = 35) => {
        return query.length > maxLen ? query.slice(0, maxLen) + "..." : query;
    };

    const getProviderIcon = (provider: string) => {
        switch (provider) {
            case "gemini": return "✨";
            case "hybrid": return "⚡";
            default: return "💻";
        }
    };

    return (
        <>
            {/* Toggle Button */}
            <button
                onClick={onToggle}
                className={`fixed top-5 left-5 z-50 p-3 rounded-xl glass-panel transition-all duration-300 hover:scale-105 ${isOpen ? 'opacity-0 pointer-events-none' : 'opacity-100'
                    }`}
                title="Open history"
            >
                <svg className="w-5 h-5 text-[var(--text-secondary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
                </svg>
            </button>

            {/* Sidebar Panel */}
            <aside
                className={`fixed top-0 left-0 h-full w-80 z-40 transition-all duration-300 ${isOpen ? "translate-x-0" : "-translate-x-full"
                    }`}
                style={{
                    background: 'var(--bg-elevated)',
                    borderRight: '1px solid var(--border)',
                }}
            >
                <div className="flex flex-col h-full">
                    {/* Header */}
                    <div className="p-5 border-b border-[var(--border)]">
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-3">
                                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-purple-500 to-cyan-500 flex items-center justify-center">
                                    <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6h4.5m4.5 0a9 9 0 11-18 0 9 9 0 0118 0z" />
                                    </svg>
                                </div>
                                <h2 className="text-base font-semibold text-[var(--text-primary)]">History</h2>
                            </div>
                            <div className="flex items-center gap-2">
                                <button
                                    onClick={fetchHistory}
                                    className="p-2 rounded-lg hover:bg-[var(--bg-muted)] transition-colors"
                                    title="Refresh"
                                >
                                    <svg className={`w-4 h-4 text-[var(--text-muted)] ${loading ? 'animate-spin' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                                        <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99" />
                                    </svg>
                                </button>
                                <button
                                    onClick={onToggle}
                                    className="p-2 rounded-lg hover:bg-[var(--bg-muted)] transition-colors"
                                    title="Close"
                                >
                                    <svg className="w-4 h-4 text-[var(--text-muted)]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                                        <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                                    </svg>
                                </button>
                            </div>
                        </div>
                    </div>

                    {/* History List */}
                    <div className="flex-1 overflow-y-auto p-3 space-y-1">
                        {loading && history.length === 0 ? (
                            <div className="flex flex-col items-center justify-center py-12 text-[var(--text-muted)]">
                                <div className="w-6 h-6 border-2 border-purple-500/30 border-t-purple-500 rounded-full animate-spin mb-3" />
                                <span className="text-sm">Loading...</span>
                            </div>
                        ) : history.length === 0 ? (
                            <div className="flex flex-col items-center justify-center py-12 text-[var(--text-muted)]">
                                <div className="w-12 h-12 rounded-xl bg-[var(--bg-muted)] flex items-center justify-center mb-3">
                                    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                                        <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
                                    </svg>
                                </div>
                                <span className="text-sm">No research yet</span>
                                <span className="text-xs mt-1">Start searching to build history</span>
                            </div>
                        ) : (
                            history.map((item) => (
                                <button
                                    key={item.dir_name}
                                    onClick={() => onSelectHistory(item)}
                                    onMouseEnter={() => setHoveredItem(item.dir_name)}
                                    onMouseLeave={() => setHoveredItem(null)}
                                    className={`w-full text-left p-3 rounded-xl transition-all duration-200 group relative ${hoveredItem === item.dir_name
                                            ? 'bg-gradient-to-r from-purple-500/10 to-cyan-500/10 border border-purple-500/20'
                                            : 'hover:bg-[var(--bg-muted)] border border-transparent'
                                        }`}
                                >
                                    <div className="flex items-start gap-3">
                                        <span className="text-lg mt-0.5">{getProviderIcon(item.provider)}</span>
                                        <div className="flex-1 min-w-0">
                                            <p className="text-sm text-[var(--text-primary)] font-medium leading-snug">
                                                {truncateQuery(item.query)}
                                            </p>
                                            <div className="flex items-center gap-2 mt-1.5 text-xs text-[var(--text-muted)]">
                                                <span>{formatDate(item.timestamp)}</span>
                                                <span className="w-1 h-1 rounded-full bg-[var(--text-muted)]" />
                                                <span>{item.source_count} sources</span>
                                            </div>
                                        </div>
                                        <button
                                            onClick={(e) => handleDelete(item.dir_name, e)}
                                            className={`p-1.5 rounded-lg transition-all duration-200 ${hoveredItem === item.dir_name
                                                    ? 'opacity-100 bg-red-500/10 hover:bg-red-500/20'
                                                    : 'opacity-0'
                                                }`}
                                            title="Delete"
                                        >
                                            <svg className="w-4 h-4 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                                                <path strokeLinecap="round" strokeLinejoin="round" d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0" />
                                            </svg>
                                        </button>
                                    </div>
                                </button>
                            ))
                        )}
                    </div>

                    {/* Footer */}
                    <div className="p-4 border-t border-[var(--border)]">
                        <div className="flex items-center gap-2 text-xs text-[var(--text-muted)]">
                            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                                <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 12.75V12A2.25 2.25 0 014.5 9.75h15A2.25 2.25 0 0121.75 12v.75m-8.69-6.44l-2.12-2.12a1.5 1.5 0 00-1.061-.44H4.5A2.25 2.25 0 002.25 6v12a2.25 2.25 0 002.25 2.25h15A2.25 2.25 0 0021.75 18V9a2.25 2.25 0 00-2.25-2.25h-5.379a1.5 1.5 0 01-1.06-.44z" />
                            </svg>
                            <span>~/Documents/socrates</span>
                        </div>
                    </div>
                </div>
            </aside>

            {/* Overlay */}
            {isOpen && (
                <div
                    className="fixed inset-0 bg-black/40 backdrop-blur-sm z-30 lg:hidden"
                    onClick={onToggle}
                />
            )}
        </>
    );
}
