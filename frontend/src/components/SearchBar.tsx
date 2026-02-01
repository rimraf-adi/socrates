"use client";

import { useState, FormEvent } from "react";

interface SearchBarProps {
    onSearch: (query: string) => void;
    isLoading: boolean;
}

export default function SearchBar({ onSearch, isLoading }: SearchBarProps) {
    const [query, setQuery] = useState("");
    const [isFocused, setIsFocused] = useState(false);

    const handleSubmit = (e: FormEvent) => {
        e.preventDefault();
        if (query.trim() && !isLoading) {
            onSearch(query);
        }
    };

    return (
        <form onSubmit={handleSubmit} className="w-full">
            <div
                className={`relative flex glass-panel overflow-hidden transition-all duration-300 ${isFocused
                        ? 'shadow-lg ring-2 ring-[var(--accent)]/30'
                        : 'shadow-md hover:shadow-lg'
                    }`}
                style={{
                    background: isFocused ? 'var(--gradient-subtle)' : undefined,
                }}
            >
                {/* Gradient border effect when focused */}
                {isFocused && (
                    <div className="absolute inset-0 rounded-2xl p-[1px] bg-gradient-to-r from-purple-500/50 via-cyan-500/50 to-purple-500/50 -z-10" />
                )}

                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onFocus={() => setIsFocused(true)}
                    onBlur={() => setIsFocused(false)}
                    placeholder="What would you like to research?"
                    className="flex-1 px-6 py-4 bg-transparent text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none text-base"
                    disabled={isLoading}
                />
                <button
                    type="submit"
                    disabled={isLoading || !query.trim()}
                    className="px-6 m-2 rounded-xl bg-gradient-to-r from-purple-500 to-cyan-500 text-white font-medium hover:from-purple-600 hover:to-cyan-600 disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-200 flex items-center gap-2 shadow-lg hover:shadow-xl hover:shadow-purple-500/20"
                >
                    {isLoading ? (
                        <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    ) : (
                        <>
                            <svg
                                className="w-5 h-5"
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                            >
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={2}
                                    d="M13 10V3L4 14h7v7l9-11h-7z"
                                />
                            </svg>
                            <span className="hidden sm:inline">Search</span>
                        </>
                    )}
                </button>
            </div>
        </form>
    );
}
