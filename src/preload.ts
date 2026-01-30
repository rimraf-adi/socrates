// See the Electron documentation for details on how to use preload scripts:
// https://www.electronjs.org/docs/latest/tutorial/process-model#preload-scripts

import { contextBridge, ipcRenderer } from 'electron';

export interface ResearchProgress {
    stage: 'decomposing' | 'researching' | 'evaluating' | 'follow-up' | 'synthesizing' | 'complete';
    currentStep: string;
    totalSteps: number;
    stepNumber: number;
    subQuestions?: string[];
    findings?: any[];
    intermediateAnswer?: string;
}

contextBridge.exposeInMainWorld('electronAPI', {
    searchSearx: (query: string) => ipcRenderer.invoke('search-searx', query),
    synthesizeAnswer: (query: string, context: string, mode: string) =>
        ipcRenderer.invoke('synthesize-answer', query, context, mode),
    deepResearch: (query: string) => ipcRenderer.invoke('deep-research', query),
    onDeepResearchProgress: (callback: (progress: ResearchProgress) => void) => {
        const handler = (_event: any, progress: ResearchProgress) => callback(progress);
        ipcRenderer.on('deep-research-progress', handler);
        // Return cleanup function
        return () => ipcRenderer.removeListener('deep-research-progress', handler);
    },
});
