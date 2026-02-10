"use client";

import { useState, useCallback, useEffect } from "react";
import SearchBar from "@/components/SearchBar";
import ResultsPane from "@/components/ResultsPane";
import AgentProgress from "@/components/AgentProgress";
import Sidebar from "@/components/Sidebar";
import SettingsModal from "@/components/SettingsModal";

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

type SearchMode = "quick" | "deep";
type Provider = "lmstudio" | "gemini" | "hybrid";
type Depth = "quick" | "standard" | "deep" | "exhaustive";

interface ResearchSettings {
  provider: Provider;
  depth: Depth;
  maxIterations: number;
}

export default function Home() {
  const [mode, setMode] = useState<SearchMode>("quick");
  const [isLoading, setIsLoading] = useState(false);
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<SearchResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<ProgressEvent | null>(null);
  const [subQuestions, setSubQuestions] = useState<string[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [researchSettings, setResearchSettings] = useState<ResearchSettings>({
    provider: "hybrid",
    depth: "standard",
    maxIterations: 10,
  });

  // Load theme preferences on mount
  useEffect(() => {
    const html = document.documentElement;
    const savedDark = localStorage.getItem("dark");
    const savedAccent = localStorage.getItem("accent");

    // Default to dark mode if not set
    if (savedDark === null || savedDark === "true") {
      html.classList.add("dark");
    } else {
      html.classList.remove("dark");
    }

    // Apply accent color
    if (savedAccent && savedAccent !== "neutral") {
      html.classList.add(`accent-${savedAccent}`);
    }

    // Load research settings
    const savedSettings = localStorage.getItem("researchSettings");
    if (savedSettings) {
      try {
        setResearchSettings(JSON.parse(savedSettings));
      } catch { }
    }
  }, []);

  const handleResearchSettingsChange = (settings: ResearchSettings) => {
    setResearchSettings(settings);
    localStorage.setItem("researchSettings", JSON.stringify(settings));
  };

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
        if (mode === "quick") {
          const response = await fetch(
            `${API_URL}/api/search?q=${encodeURIComponent(query)}&provider=${researchSettings.provider}`
          );
          if (!response.ok) throw new Error(`Search failed: ${response.status}`);
          const data = await response.json();
          setAnswer(data.answer);
          setSources(data.sources);
        } else {
          const eventSource = new EventSource(
            `${API_URL}/api/research?q=${encodeURIComponent(query)}&provider=${researchSettings.provider}&depth=${researchSettings.depth}&max_iterations=${researchSettings.maxIterations}`
          );

          eventSource.onmessage = (event) => {
            try {
              const data: ProgressEvent = JSON.parse(event.data);
              if (data.type === "progress") {
                setProgress(data);
                if (data.sub_questions) setSubQuestions(data.sub_questions);
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
              console.error("Failed to parse SSE:", e);
            }
          };

          eventSource.onerror = () => {
            setError("Connection lost");
            setIsLoading(false);
            eventSource.close();
          };
          return;
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unknown error");
      } finally {
        if (mode === "quick") setIsLoading(false);
      }
    },
    [mode, researchSettings]
  );

  const handleSelectHistory = async (item: HistoryItem) => {
    try {
      const response = await fetch(`${API_URL}/api/history/${encodeURIComponent(item.dir_name)}`);
      if (response.ok) {
        const data = await response.json();
        const content = data.content || "";
        const answerMatch = content.match(/## Research Answer\n\n([\s\S]*?)(?=\n---|$)/);
        setAnswer(answerMatch ? answerMatch[1].trim() : content);
        setSources(data.sources || []);
        setMode("deep");
        setSidebarOpen(false);
      }
    } catch (error) {
      console.error("Failed to load history:", error);
    }
  };

  return (
    <>
      <Sidebar
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
        onSelectHistory={handleSelectHistory}
      />

      <SettingsModal
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        researchSettings={researchSettings}
        onResearchSettingsChange={handleResearchSettingsChange}
      />

      <main className={`min-h-screen bg-[var(--bg-base)] transition-all ${sidebarOpen ? 'lg:pl-80' : ''}`}>
        {/* Top Bar */}
        <div className="fixed top-0 left-0 right-0 z-30 p-4 flex items-center justify-between">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 rounded-lg hover:bg-[var(--bg-muted)] transition-colors"
          >
            <svg className="w-5 h-5 text-[var(--text-muted)]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
            </svg>
          </button>
          <button
            onClick={() => setSettingsOpen(true)}
            className="p-2 rounded-lg hover:bg-[var(--bg-muted)] transition-colors"
          >
            <svg className="w-5 h-5 text-[var(--text-muted)]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.324.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 011.37.49l1.296 2.247a1.125 1.125 0 01-.26 1.431l-1.003.827c-.293.24-.438.613-.431.992a6.759 6.759 0 010 .255c-.007.378.138.75.43.99l1.005.828c.424.35.534.954.26 1.43l-1.298 2.247a1.125 1.125 0 01-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.57 6.57 0 01-.22.128c-.331.183-.581.495-.644.869l-.213 1.28c-.09.543-.56.941-1.11.941h-2.594c-.55 0-1.02-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 01-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 01-1.369-.49l-1.297-2.247a1.125 1.125 0 01.26-1.431l1.004-.827c.292-.24.437-.613.43-.992a6.932 6.932 0 010-.255c.007-.378-.138-.75-.43-.99l-1.004-.828a1.125 1.125 0 01-.26-1.43l1.297-2.247a1.125 1.125 0 011.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.087.22-.128.332-.183.582-.495.644-.869l.214-1.281z" />
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </button>
        </div>

        {/* Main Content */}
        <div className="max-w-2xl mx-auto px-6 pt-24 pb-12">
          {/* Header - Only show when no results */}
          {!answer && !isLoading && (
            <header className="text-center mb-12 pt-12">
              <h1 className="text-3xl font-medium tracking-tight mb-3">socrates</h1>
              <p className="text-[var(--text-muted)] text-sm">research anything</p>
            </header>
          )}

          {/* Mode Toggle */}
          <div className="flex justify-center mb-6">
            <div className="inline-flex p-1 rounded-lg bg-[var(--bg-muted)]">
              <button
                onClick={() => setMode("quick")}
                className={`px-4 py-2 rounded-md text-sm transition-colors ${mode === "quick"
                  ? "bg-[var(--bg-base)] text-[var(--text-primary)] shadow-sm"
                  : "text-[var(--text-muted)] hover:text-[var(--text-secondary)]"
                  }`}
              >
                Quick
              </button>
              <button
                onClick={() => setMode("deep")}
                className={`px-4 py-2 rounded-md text-sm transition-colors ${mode === "deep"
                  ? "bg-[var(--bg-base)] text-[var(--text-primary)] shadow-sm"
                  : "text-[var(--text-muted)] hover:text-[var(--text-secondary)]"
                  }`}
              >
                Deep Research
              </button>
            </div>
          </div>

          {/* Search */}
          <div className="mb-8">
            <SearchBar onSearch={handleSearch} isLoading={isLoading} />
          </div>

          {/* Progress */}
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
        </div>
      </main>
    </>
  );
}
