"use client";

import { useState, useEffect, useCallback } from "react";
import SearchBar from "@/components/SearchBar";
import ModeToggle from "@/components/ModeToggle";
import ResultsPane from "@/components/ResultsPane";
import AgentProgress from "@/components/AgentProgress";

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

export default function Home() {
  const [mode, setMode] = useState<"simple" | "deep">("simple");
  const [isLoading, setIsLoading] = useState(false);
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<SearchResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<ProgressEvent | null>(null);
  const [subQuestions, setSubQuestions] = useState<string[]>([]);

  const handleSearch = useCallback(
    async (query: string) => {
      if (!query.trim()) return;

      setIsLoading(true);
      setAnswer("");
      setSources([]);
      setError(null);
      setProgress(null);
      setSubQuestions([]);

      try {
        if (mode === "simple") {
          // Simple search - REST API
          const response = await fetch(
            `${API_URL}/api/search?q=${encodeURIComponent(query)}`
          );

          if (!response.ok) {
            throw new Error(`Search failed: ${response.status}`);
          }

          const data = await response.json();
          setAnswer(data.answer);
          setSources(data.sources);
        } else {
          // Deep research - SSE streaming
          const eventSource = new EventSource(
            `${API_URL}/api/research?q=${encodeURIComponent(query)}`
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
    [mode]
  );

  return (
    <main className="min-h-screen bg-gradient-to-b from-gray-950 to-gray-900 text-white">
      <div className="max-w-4xl mx-auto px-4 py-12">
        {/* Header */}
        <header className="text-center mb-12">
          <h1 className="text-5xl font-bold bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent mb-2">
            Socrates
          </h1>
          <p className="text-gray-400">Open-Source AI Research Assistant</p>
        </header>

        {/* Search Controls */}
        <div className="space-y-6 mb-8">
          <ModeToggle mode={mode} onModeChange={setMode} />
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
        <footer className="text-center text-gray-500 text-sm mt-12">
          Powered by LangGraph + Groq + SearXNG
        </footer>
      </div>
    </main>
  );
}
