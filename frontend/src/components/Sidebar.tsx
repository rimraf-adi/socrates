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

    const fetchHistory = async () => {
        setLoading(true);
        try {
            const response = await fetch(`${API_URL}/api/history`);
            if (response.ok) setHistory(await response.json());
        } catch (error) {
            console.error("Failed to fetch history:", error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        if (isOpen) fetchHistory();
    }, [isOpen]);

    const handleDelete = async (dirName: string, e: React.MouseEvent) => {
        e.stopPropagation();
        try {
            const response = await fetch(`${API_URL}/api/history/${encodeURIComponent(dirName)}`, { method: "DELETE" });
            if (response.ok) setHistory((prev) => prev.filter((item) => item.dir_name !== dirName));
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

    return (
        <>
            {/* Sidebar */}
            <aside
                className={`fixed top-0 left-0 h-full w-80 z-40 bg-[var(--bg-elevated)] border-r border-[var(--border)] transition-transform duration-200 ${isOpen ? "translate-x-0" : "-translate-x-full"
                    }`}
            >
                <div className="flex flex-col h-full">
                    {/* Header */}
                    <div className="p-4 border-b border-[var(--border)] flex items-center justify-between">
                        <span className="text-sm font-medium">History</span>
                        <div className="flex gap-1">
                            <button onClick={fetchHistory} className="p-1.5 rounded hover:bg-[var(--bg-muted)]">
                                <svg className={`w-4 h-4 text-[var(--text-muted)] ${loading ? 'animate-spin' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99" />
                                </svg>
                            </button>
                            <button onClick={onToggle} className="p-1.5 rounded hover:bg-[var(--bg-muted)]">
                                <svg className="w-4 h-4 text-[var(--text-muted)]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                                </svg>
                            </button>
                        </div>
                    </div>

                    {/* List */}
                    <div className="flex-1 overflow-y-auto p-2">
                        {loading && history.length === 0 ? (
                            <p className="text-center text-[var(--text-muted)] text-xs py-8">Loading...</p>
                        ) : history.length === 0 ? (
                            <p className="text-center text-[var(--text-muted)] text-xs py-8">No history yet</p>
                        ) : (
                            history.map((item) => (
                                <button
                                    key={item.dir_name}
                                    onClick={() => onSelectHistory(item)}
                                    className="w-full text-left p-3 rounded-lg hover:bg-[var(--bg-muted)] transition-colors group"
                                >
                                    <p className="text-sm text-[var(--text-primary)] truncate">
                                        {item.query}
                                    </p>
                                    <div className="flex items-center justify-between mt-1">
                                        <span className="text-xs text-[var(--text-muted)]">{formatDate(item.timestamp)}</span>
                                        <button
                                            onClick={(e) => handleDelete(item.dir_name, e)}
                                            className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-red-500/10 transition-opacity"
                                        >
                                            <svg className="w-3.5 h-3.5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                                                <path strokeLinecap="round" strokeLinejoin="round" d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0" />
                                            </svg>
                                        </button>
                                    </div>
                                </button>
                            ))
                        )}
                    </div>

                    {/* Footer */}
                    <div className="p-3 border-t border-[var(--border)]">
                        <p className="text-xs text-[var(--text-muted)]">~/Documents/socrates</p>
                    </div>
                </div>
            </aside>

            {/* Overlay */}
            {isOpen && <div className="fixed inset-0 bg-black/30 z-30 lg:hidden" onClick={onToggle} />}
        </>
    );
}
