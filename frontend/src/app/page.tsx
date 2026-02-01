"use client";

import { useState, useCallback } from "react";
import SearchBar from "@/components/SearchBar";
import ModeToggle from "@/components/ModeToggle";
import ModelSelector from "@/components/ModelSelector";
import ProviderSelector from "@/components/ProviderSelector";
import ResearchSettings from "@/components/ResearchSettings";
import ResultsPane from "@/components/ResultsPane";
import AgentProgress from "@/components/AgentProgress";
import ThemeControls from "@/components/ThemeControls";
import Sidebar from "@/components/Sidebar";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface SearchResult {
  title: string;
  url: string;
  snippet: string;
}

interface ProgressEvent {
  type: string;
  node?: string;
  status?: string;
  message?: string;
  sub_questions?: string[];
  iteration?: number;
  answer?: string;
  sources?: SearchResult[];
}

interface HistoryItem {
  id: string;
  query: string;
  provider: string;
  depth: string;
  timestamp: string;
  source_count: number;
  dir_name: string;
}

export default function Home() {
  const [mode, setMode] = useState<"simple" | "deep">("simple");
  const [provider, setProvider] = useState<"lmstudio" | "gemini" | "hybrid">("hybrid");
  const [selectedModel, setSelectedModel] = useState("");
  const [depth, setDepth] = useState<"quick" | "standard" | "deep" | "exhaustive">("standard");
  const [maxIterations, setMaxIterations] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<SearchResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<ProgressEvent | null>(null);
  const [subQuestions, setSubQuestions] = useState<string[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const handleSearch = useCallback(
    async (query: string) => {
      if (!query.trim()) return;

      setIsLoading(true);
      setAnswer("");
      setSources([]);
      setError(null);
      setProgress(null);
      setSubQuestions([]);

      const modelParam = selectedModel ? `&model=${encodeURIComponent(selectedModel)}` : "";
      const providerParam = `&provider=${provider}`;
      const depthParam = `&depth=${depth}`;
      const iterationsParam = maxIterations ? `&max_iterations=${maxIterations}` : "";

      try {
        if (mode === "simple") {
          const response = await fetch(
            `${API_URL}/api/search?q=${encodeURIComponent(query)}${modelParam}${providerParam}`
          );

          if (!response.ok) {
            throw new Error(`Search failed: ${response.status}`);
          }

          const data = await response.json();
          setAnswer(data.answer);
          setSources(data.sources);
        } else {
          const eventSource = new EventSource(
            `${API_URL}/api/research?q=${encodeURIComponent(query)}${modelParam}${providerParam}${depthParam}${iterationsParam}`
          );

          eventSource.onmessage = (event) => {
            try {
              const data: ProgressEvent = JSON.parse(event.data);

              if (data.type === "progress") {
                setProgress(data);
                if (data.sub_questions) {
                  setSubQuestions(data.sub_questions);
                }
              } else if (data.type === "complete") {
                setAnswer(data.answer || "");
                setSources(data.sources || []);
                setProgress(null);
                setIsLoading(false);
                eventSource.close();
              } else if (data.type === "error") {
                setError(data.message || "Research failed");
                setIsLoading(false);
                eventSource.close();
              }
            } catch (e) {
              console.error("Failed to parse SSE event:", e);
            }
          };

          eventSource.onerror = () => {
            setError("Connection to research agent lost");
            setIsLoading(false);
            eventSource.close();
          };
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unknown error");
      } finally {
        if (mode === "simple") {
          setIsLoading(false);
        }
      }
    },
    [mode, selectedModel, provider, depth, maxIterations]
  );

  const handleSelectHistory = async (item: HistoryItem) => {
    // Load the saved research
    try {
      const response = await fetch(`${API_URL}/api/history/${encodeURIComponent(item.dir_name)}`);
      if (response.ok) {
        const data = await response.json();
        // Extract answer from markdown content (skip the header lines)
        const content = data.content || "";
        const answerMatch = content.match(/## Research Answer\n\n([\s\S]*?)(?=\n---|\n## |$)/);
        const answerText = answerMatch ? answerMatch[1].trim() : content;

        setAnswer(answerText);
        setSources(data.sources || []);
        setSubQuestions(data.metadata?.sub_questions || []);
        setMode("deep");
        setSidebarOpen(false);
      }
    } catch (error) {
      console.error("Failed to load history:", error);
    }
  };

  return (
    <>
      {/* Sidebar */}
      <Sidebar
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
        onSelectHistory={handleSelectHistory}
      />

      <main className={`min-h-screen bg-[var(--bg-base)] text-[var(--text-primary)] transition-all duration-300 ${sidebarOpen ? 'lg:pl-80' : ''}`}>
        <div className="max-w-3xl mx-auto px-6 py-12">
          {/* Header */}
          <header className="text-center mb-12">
            <div className="inline-block mb-4">
              <div className="w-14 h-14 mx-auto rounded-2xl bg-gradient-to-br from-purple-500 to-cyan-500 flex items-center justify-center shadow-lg shadow-purple-500/30 animate-float">
                <svg className="w-7 h-7 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
                </svg>
              </div>
            </div>
            <h1 className="text-5xl font-bold tracking-tight mb-3">
              <span className="gradient-text">socrates</span>
            </h1>
            <p className="text-base text-[var(--text-secondary)]">
              AI-powered deep research assistant
            </p>
          </header>

          {/* Search Controls */}
          <div className="space-y-5 mb-10">
            <div className="flex gap-4 justify-center flex-wrap">
              <ProviderSelector provider={provider} onProviderChange={setProvider} />
              <ModeToggle mode={mode} onModeChange={setMode} />
            </div>
            {provider === "lmstudio" && (
              <div className="flex justify-center">
                <ModelSelector selectedModel={selectedModel} onModelChange={setSelectedModel} />
              </div>
            )}
            {mode === "deep" && (
              <ResearchSettings
                depth={depth}
                maxIterations={maxIterations}
                onDepthChange={setDepth}
                onMaxIterationsChange={setMaxIterations}
              />
            )}
            <SearchBar onSearch={handleSearch} isLoading={isLoading} />
          </div>

          {/* Progress (Deep Research) */}
          {mode === "deep" && progress && (
            <AgentProgress progress={progress} subQuestions={subQuestions} />
          )}

          {/* Results */}
          <ResultsPane
            answer={answer}
            sources={sources}
            isLoading={isLoading && !progress}
            error={error}
            isDeepResearch={mode === "deep"}
          />

          {/* Footer */}
          <footer className="text-center text-[var(--text-muted)] text-xs mt-16 tracking-wide">
            langgraph + lmstudio/gemini + searxng
          </footer>
        </div>

        {/* Theme Controls */}
        <ThemeControls />
      </main>
    </>
  );
}



