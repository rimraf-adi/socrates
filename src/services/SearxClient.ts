export interface SearchResult {
    title: string;
    url: string;
    snippet: string;
}

export interface ResearchFinding {
    subQuestion: string;
    sources: SearchResult[];
    keyPoints: string[];
}

export interface ResearchProgress {
    stage: 'thinking' | 'searching' | 'analyzing' | 'synthesizing' | 'complete' | 'decomposing' | 'researching' | 'evaluating' | 'follow-up';
    currentStep: string;
    thought?: string;
    action?: string;
    observation?: string;
    subQuestions?: string[];
    findings?: ResearchFinding[];
    intermediateAnswer?: string;
    totalSteps?: number;
    stepNumber?: number;
}

export interface DeepResearchResult {
    answer: string;
    sources: SearchResult[];
    thoughtProcess?: string[];
    subQuestions?: string[];
    findings?: ResearchFinding[];
    iterations: number;
}

declare global {
    interface Window {
        electronAPI: {
            searchSearx: (query: string) => Promise<SearchResult[]>;
            synthesizeAnswer: (query: string, context: string, mode: string) => Promise<string>;
            deepResearch: (query: string) => Promise<DeepResearchResult>;
            onDeepResearchProgress: (callback: (progress: ResearchProgress) => void) => () => void;
        };
    }
}

export async function searchSearx(query: string): Promise<SearchResult[]> {
    return window.electronAPI.searchSearx(query);
}

export async function deepResearch(query: string): Promise<DeepResearchResult> {
    return window.electronAPI.deepResearch(query);
}

export function onDeepResearchProgress(callback: (progress: ResearchProgress) => void): () => void {
    return window.electronAPI.onDeepResearchProgress(callback);
}
