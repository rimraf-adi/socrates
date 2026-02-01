"use client";

import { useState, useEffect, useRef } from "react";

interface ModelSelectorProps {
    selectedModel: string;
    onModelChange: (model: string) => void;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function ModelSelector({ selectedModel, onModelChange }: ModelSelectorProps) {
    const [models, setModels] = useState<string[]>([]);
    const [isOpen, setIsOpen] = useState(false);
    const [isAdding, setIsAdding] = useState(false);
    const [newModelId, setNewModelId] = useState("");
    const [isLoading, setIsLoading] = useState(true);
    const dropdownRef = useRef<HTMLDivElement>(null);

    // Fetch models on mount
    useEffect(() => {
        fetchModels();
    }, []);

    // Close dropdown when clicking outside
    useEffect(() => {
        function handleClickOutside(event: MouseEvent) {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
                setIsOpen(false);
                setIsAdding(false);
            }
        }
        document.addEventListener("mousedown", handleClickOutside);
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, []);

    const fetchModels = async () => {
        try {
            const response = await fetch(`${API_URL}/api/models`);
            if (response.ok) {
                const data = await response.json();
                setModels(data.models);
                if (data.models.length > 0 && !selectedModel) {
                    onModelChange(data.models[0]);
                }
            }
        } catch (error) {
            console.error("Failed to fetch models:", error);
        } finally {
            setIsLoading(false);
        }
    };

    const addModel = async () => {
        if (!newModelId.trim()) return;

        try {
            const response = await fetch(`${API_URL}/api/models`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model_id: newModelId.trim() }),
            });

            if (response.ok) {
                const data = await response.json();
                setModels(data.models);
                setNewModelId("");
                setIsAdding(false);
                onModelChange(newModelId.trim());
            }
        } catch (error) {
            console.error("Failed to add model:", error);
        }
    };

    const deleteModel = async (modelId: string) => {
        try {
            const response = await fetch(`${API_URL}/api/models/${encodeURIComponent(modelId)}`, {
                method: "DELETE",
            });

            if (response.ok) {
                const data = await response.json();
                setModels(data.models);
                if (selectedModel === modelId && data.models.length > 0) {
                    onModelChange(data.models[0]);
                }
            }
        } catch (error) {
            console.error("Failed to delete model:", error);
        }
    };

    const displayName = (modelId: string) => {
        // Shorten model names for display
        const parts = modelId.split("/");
        return parts[parts.length - 1] || modelId;
    };

    return (
        <div ref={dropdownRef} className="relative">
            {/* Selector Button */}
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="glass-panel px-4 py-2.5 flex items-center gap-3 hover:bg-[var(--bg-muted)] transition-colors min-w-[200px]"
                disabled={isLoading}
            >
                <svg className="w-4 h-4 text-[var(--text-muted)]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 3v1.5M4.5 8.25H3m18 0h-1.5M4.5 12H3m18 0h-1.5m-15 3.75H3m18 0h-1.5M8.25 19.5V21M12 3v1.5m0 15V21m3.75-18v1.5m0 15V21m-9-1.5h10.5a2.25 2.25 0 002.25-2.25V6.75a2.25 2.25 0 00-2.25-2.25H6.75A2.25 2.25 0 004.5 6.75v10.5a2.25 2.25 0 002.25 2.25zm.75-12h9v9h-9v-9z" />
                </svg>
                <span className="text-sm text-[var(--text-secondary)] flex-1 text-left truncate">
                    {isLoading ? "Loading..." : (selectedModel ? displayName(selectedModel) : "Select model")}
                </span>
                <svg className={`w-4 h-4 text-[var(--text-muted)] transition-transform ${isOpen ? "rotate-180" : ""}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
                </svg>
            </button>

            {/* Dropdown */}
            {isOpen && (
                <div className="absolute top-full left-0 right-0 mt-2 glass-panel overflow-hidden z-50">
                    {/* Model List */}
                    <div className="max-h-48 overflow-y-auto">
                        {models.length === 0 ? (
                            <div className="px-4 py-3 text-sm text-[var(--text-muted)]">
                                No models configured
                            </div>
                        ) : (
                            models.map((model) => (
                                <div
                                    key={model}
                                    className={`px-4 py-2.5 flex items-center gap-2 hover:bg-[var(--bg-muted)] cursor-pointer group transition-colors ${selectedModel === model ? "bg-[var(--bg-muted)]" : ""
                                        }`}
                                    onClick={() => {
                                        onModelChange(model);
                                        setIsOpen(false);
                                    }}
                                >
                                    <span className="text-sm text-[var(--text-secondary)] flex-1 truncate" title={model}>
                                        {displayName(model)}
                                    </span>
                                    {selectedModel === model && (
                                        <svg className="w-4 h-4 text-[var(--accent)]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                            <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
                                        </svg>
                                    )}
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            deleteModel(model);
                                        }}
                                        className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-red-500/20 text-[var(--text-muted)] hover:text-red-400 transition-all"
                                    >
                                        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                                            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                                        </svg>
                                    </button>
                                </div>
                            ))
                        )}
                    </div>

                    {/* Divider */}
                    <div className="border-t border-[var(--border-default)]" />

                    {/* Add New Model */}
                    {isAdding ? (
                        <div className="p-3 flex gap-2">
                            <input
                                type="text"
                                value={newModelId}
                                onChange={(e) => setNewModelId(e.target.value)}
                                onKeyDown={(e) => e.key === "Enter" && addModel()}
                                placeholder="Enter model ID..."
                                className="flex-1 bg-[var(--bg-base)] border border-[var(--border-default)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--accent)]"
                                autoFocus
                            />
                            <button
                                onClick={addModel}
                                className="px-3 py-2 bg-[var(--accent)] text-white rounded-lg text-sm font-medium hover:opacity-90 transition-opacity"
                            >
                                Add
                            </button>
                        </div>
                    ) : (
                        <button
                            onClick={() => setIsAdding(true)}
                            className="w-full px-4 py-2.5 flex items-center gap-2 hover:bg-[var(--bg-muted)] transition-colors text-[var(--accent)]"
                        >
                            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                                <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
                            </svg>
                            <span className="text-sm">Add model</span>
                        </button>
                    )}
                </div>
            )}
        </div>
    );
}
