import React, { useState, useCallback, useEffect } from 'react';
import SearchBar from './components/SearchBar';
import ResultsPane from './components/ResultsPane';
import ModeToggle from './components/ModeToggle';
import ResearchProgress from './components/ResearchProgress';
import { searchSearx, SearchResult, deepResearch, onDeepResearchProgress, ResearchProgress as ResearchProgressType, ResearchFinding } from './services/SearxClient';
import { synthesizeAnswer } from './services/GroqClient';

export type SearchMode = 'simple' | 'deep';

interface SearchState {
    isLoading: boolean;
    answer: string;
    sources: SearchResult[];
    error: string | null;
    // Deep research specific
    subQuestions: string[];
    findings: ResearchFinding[];
    researchProgress: ResearchProgressType | null;
}

const App: React.FC = () => {
    const [mode, setMode] = useState<SearchMode>('simple');
    const [searchState, setSearchState] = useState<SearchState>({
        isLoading: false,
        answer: '',
        sources: [],
        error: null,
        subQuestions: [],
        findings: [],
        researchProgress: null,
    });

    // Setup progress listener for deep research
    useEffect(() => {
        const cleanup = onDeepResearchProgress((progress) => {
            setSearchState(prev => ({
                ...prev,
                researchProgress: progress,
                subQuestions: progress.subQuestions || prev.subQuestions,
                findings: progress.findings || prev.findings,
            }));
        });

        return cleanup;
    }, []);

    const handleSearch = useCallback(async (query: string) => {
        if (!query.trim()) return;

        setSearchState({
            isLoading: true,
            answer: '',
            sources: [],
            error: null,
            subQuestions: [],
            findings: [],
            researchProgress: null,
        });

        try {
            if (mode === 'simple') {
                // Simple search - single pass
                const searchResults = await searchSearx(query);

                if (searchResults.length === 0) {
                    setSearchState(prev => ({
                        ...prev,
                        isLoading: false,
                        answer: 'No results found for your query.',
                    }));
                    return;
                }

                const context = searchResults.map(r => `Source: ${r.title}\nURL: ${r.url}\nContent: ${r.snippet}`).join('\n\n');
                const answer = await synthesizeAnswer(query, context, mode);

                setSearchState(prev => ({
                    ...prev,
                    isLoading: false,
                    answer,
                    sources: searchResults,
                }));
            } else {
                // Deep research - agentic multi-step
                const result = await deepResearch(query);

                setSearchState(prev => ({
                    ...prev,
                    isLoading: false,
                    answer: result.answer,
                    sources: result.sources,
                    subQuestions: result.subQuestions,
                    findings: result.findings,
                    researchProgress: null,
                }));
            }
        } catch (err) {
            setSearchState(prev => ({
                ...prev,
                isLoading: false,
                error: err instanceof Error ? err.message : 'An unknown error occurred',
            }));
        }
    }, [mode]);

    return (
        <div className="app-container">
            <header className="app-header">
                <h1 className="app-title">Socrates</h1>
                <p className="app-subtitle">Open-Source AI-Powered Search</p>
            </header>

            <main className="app-main">
                <div className="search-controls">
                    <ModeToggle mode={mode} onModeChange={setMode} />
                    <SearchBar onSearch={handleSearch} isLoading={searchState.isLoading} />
                </div>

                {/* Show research progress for deep mode */}
                {mode === 'deep' && searchState.researchProgress && (
                    <ResearchProgress progress={searchState.researchProgress} />
                )}

                <ResultsPane
                    answer={searchState.answer}
                    sources={searchState.sources}
                    isLoading={searchState.isLoading && !searchState.researchProgress}
                    error={searchState.error}
                    subQuestions={searchState.subQuestions}
                    findings={searchState.findings}
                    isDeepResearch={mode === 'deep'}
                />
            </main>

            <footer className="app-footer">
                <p>Powered by SearXNG & Groq</p>
            </footer>
        </div>
    );
};

export default App;
