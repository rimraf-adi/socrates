"use client";

import { useState, useRef, useEffect, FormEvent, KeyboardEvent } from "react";

interface SearchBarProps {
    onSearch: (query: string) => void;
    isLoading: boolean;
}

export default function SearchBar({ onSearch, isLoading }: SearchBarProps) {
    const [query, setQuery] = useState("");
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    // Auto-resize textarea
    useEffect(() => {
        const textarea = textareaRef.current;
        if (textarea) {
            textarea.style.height = "auto";
            textarea.style.height = Math.min(textarea.scrollHeight, 200) + "px";
        }
    }, [query]);

    const handleSubmit = (e?: FormEvent) => {
        e?.preventDefault();
        if (query.trim() && !isLoading) {
            onSearch(query);
        }
    };

    const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
        // Enter without Shift = Submit
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
        }
        // Shift+Enter = newline (default behavior)
    };

    return (
        <form onSubmit={handleSubmit} className="w-full">
            <div className="flex items-end gap-3 p-3 rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] focus-within:border-[var(--text-muted)] transition-colors">
                <svg className="w-5 h-5 text-[var(--text-muted)] flex-shrink-0 mb-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                <textarea
                    ref={textareaRef}
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Ask anything..."
                    rows={1}
                    className="flex-1 bg-transparent text-[var(--text-primary)] placeholder-[var(--text-muted)] resize-none text-sm leading-relaxed min-h-[24px] max-h-[200px]"
                    disabled={isLoading}
                />
                {isLoading ? (
                    <div className="w-5 h-5 border-2 border-[var(--text-muted)] border-t-[var(--text-primary)] rounded-full animate-spin flex-shrink-0" />
                ) : query.trim() && (
                    <button
                        type="submit"
                        className="p-1.5 rounded-lg bg-[var(--text-primary)] text-[var(--bg-base)] hover:opacity-80 transition-opacity flex-shrink-0"
                    >
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
                        </svg>
                    </button>
                )}
            </div>
        </form>
    );
}
