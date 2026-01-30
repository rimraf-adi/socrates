// Agentic Deep Research Engine
// Implements multi-step research with query decomposition, iterative search, and synthesis

const SEARXNG_BASE_URL = process.env.VITE_SEARXNG_BASE_URL || 'http://localhost:8080';
const GROQ_API_KEY = process.env.VITE_GROQ_API_KEY || '';

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
    stage: 'decomposing' | 'researching' | 'evaluating' | 'follow-up' | 'synthesizing' | 'complete';
    currentStep: string;
    totalSteps: number;
    stepNumber: number;
    subQuestions?: string[];
    findings?: ResearchFinding[];
    intermediateAnswer?: string;
}

export interface DeepResearchResult {
    answer: string;
    sources: SearchResult[];
    subQuestions: string[];
    findings: ResearchFinding[];
    iterations: number;
}

const MAX_ITERATIONS = 3;
const MAX_SOURCES_PER_SUBQ = 5;

async function callGroq(messages: Array<{ role: string; content: string }>, maxTokens: number = 2048): Promise<string> {
    const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${GROQ_API_KEY}`,
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            model: 'llama-3.3-70b-versatile',
            messages,
            temperature: 0.3,
            max_tokens: maxTokens,
        }),
    });

    if (!response.ok) {
        throw new Error(`Groq API failed: ${response.status}`);
    }

    const data = await response.json();
    return data.choices[0]?.message?.content || '';
}

async function searchSearx(query: string): Promise<SearchResult[]> {
    const params = new URLSearchParams({
        q: query,
        format: 'json',
        categories: 'general',
    });

    const response = await fetch(`${SEARXNG_BASE_URL}/search?${params.toString()}`);

    if (!response.ok) {
        throw new Error(`SearXNG search failed: ${response.status}`);
    }

    const data = await response.json();
    return data.results.slice(0, MAX_SOURCES_PER_SUBQ).map((result: any) => ({
        title: result.title || 'Untitled',
        url: result.url,
        snippet: result.content || '',
    }));
}

export async function decomposeQuery(query: string): Promise<string[]> {
    const response = await callGroq([
        {
            role: 'system',
            content: `You are a research planner. Break down the user's question into 3-5 specific sub-questions that together will provide a comprehensive answer. 

Output ONLY a JSON array of strings, nothing else. Example:
["What is X?", "How does X work?", "What are the benefits of X?"]`
        },
        {
            role: 'user',
            content: query
        }
    ], 512);

    try {
        // Extract JSON array from response
        const match = response.match(/\[[\s\S]*\]/);
        if (match) {
            return JSON.parse(match[0]);
        }
        // Fallback: treat as the original query
        return [query];
    } catch {
        return [query];
    }
}

export async function extractKeyPoints(subQuestion: string, sources: SearchResult[]): Promise<string[]> {
    if (sources.length === 0) return [];

    const context = sources.map((s, i) => `[${i + 1}] ${s.title}: ${s.snippet}`).join('\n\n');

    const response = await callGroq([
        {
            role: 'system',
            content: `You are a research analyst. Extract 3-5 key factual points from the search results that answer the question. Be specific and cite sources using [1], [2], etc.

Output ONLY a JSON array of strings, nothing else.`
        },
        {
            role: 'user',
            content: `Question: ${subQuestion}\n\nSearch Results:\n${context}`
        }
    ], 1024);

    try {
        const match = response.match(/\[[\s\S]*\]/);
        if (match) {
            return JSON.parse(match[0]);
        }
        return [response];
    } catch {
        return [response];
    }
}

export async function evaluateCompleteness(query: string, findings: ResearchFinding[]): Promise<{ complete: boolean; gaps: string[] }> {
    const findingsSummary = findings.map(f =>
        `Sub-question: ${f.subQuestion}\nKey points: ${f.keyPoints.join('; ')}`
    ).join('\n\n');

    const response = await callGroq([
        {
            role: 'system',
            content: `You are a research evaluator. Given the original question and research findings, determine if the research is complete enough for a comprehensive answer.

Output ONLY valid JSON in this format:
{"complete": true/false, "gaps": ["missing topic 1", "missing topic 2"]}`
        },
        {
            role: 'user',
            content: `Original Question: ${query}\n\nFindings:\n${findingsSummary}`
        }
    ], 512);

    try {
        const match = response.match(/\{[\s\S]*\}/);
        if (match) {
            return JSON.parse(match[0]);
        }
        return { complete: true, gaps: [] };
    } catch {
        return { complete: true, gaps: [] };
    }
}

export async function synthesizeReport(query: string, findings: ResearchFinding[]): Promise<string> {
    const allSources: SearchResult[] = [];
    const findingsContext = findings.map((f, idx) => {
        const sourceOffset = allSources.length;
        f.sources.forEach(s => allSources.push(s));

        return `## ${f.subQuestion}\n${f.keyPoints.map((p, i) =>
            `- ${p.replace(/\[(\d+)\]/g, (_, n) => `[${parseInt(n) + sourceOffset}]`)}`
        ).join('\n')}`;
    }).join('\n\n');

    const response = await callGroq([
        {
            role: 'system',
            content: `You are an expert research analyst writing a comprehensive report. Structure your response with:

1. **Executive Summary** - 2-3 sentence overview
2. **Detailed Analysis** - Organized by topic, using the research findings
3. **Key Takeaways** - Bullet points of main conclusions
4. **Open Questions** - Any remaining uncertainties

Use markdown formatting. Cite sources using [1], [2], etc. based on the source numbers provided.`
        },
        {
            role: 'user',
            content: `Question: ${query}\n\nResearch Findings:\n${findingsContext}\n\nSources:\n${allSources.map((s, i) => `[${i + 1}] ${s.title} - ${s.url}`).join('\n')}`
        }
    ], 4096);

    return response;
}

export async function conductDeepResearch(
    query: string,
    onProgress: (progress: ResearchProgress) => void
): Promise<DeepResearchResult> {
    const allFindings: ResearchFinding[] = [];
    const allSources: SearchResult[] = [];
    let iterations = 0;

    // Step 1: Decompose query
    onProgress({
        stage: 'decomposing',
        currentStep: 'Breaking down your question into sub-questions...',
        totalSteps: 4,
        stepNumber: 1,
    });

    const subQuestions = await decomposeQuery(query);

    onProgress({
        stage: 'decomposing',
        currentStep: `Identified ${subQuestions.length} research areas`,
        totalSteps: 4,
        stepNumber: 1,
        subQuestions,
    });

    // Step 2: Research each sub-question
    let questionsToResearch = [...subQuestions];

    while (iterations < MAX_ITERATIONS && questionsToResearch.length > 0) {
        iterations++;

        for (let i = 0; i < questionsToResearch.length; i++) {
            const subQ = questionsToResearch[i];

            onProgress({
                stage: 'researching',
                currentStep: `Researching: "${subQ.substring(0, 50)}..."`,
                totalSteps: questionsToResearch.length,
                stepNumber: i + 1,
                subQuestions,
                findings: allFindings,
            });

            // Search
            const sources = await searchSearx(subQ);
            allSources.push(...sources);

            // Extract key points
            const keyPoints = await extractKeyPoints(subQ, sources);

            allFindings.push({
                subQuestion: subQ,
                sources,
                keyPoints,
            });
        }

        // Step 3: Evaluate completeness
        onProgress({
            stage: 'evaluating',
            currentStep: 'Evaluating research completeness...',
            totalSteps: 4,
            stepNumber: 3,
            subQuestions,
            findings: allFindings,
        });

        const evaluation = await evaluateCompleteness(query, allFindings);

        if (evaluation.complete || iterations >= MAX_ITERATIONS) {
            break;
        }

        // Step 4: Generate follow-up questions for gaps
        onProgress({
            stage: 'follow-up',
            currentStep: `Found ${evaluation.gaps.length} knowledge gaps, researching more...`,
            totalSteps: 4,
            stepNumber: 4,
            subQuestions: [...subQuestions, ...evaluation.gaps],
            findings: allFindings,
        });

        questionsToResearch = evaluation.gaps;
    }

    // Step 5: Synthesize final report
    onProgress({
        stage: 'synthesizing',
        currentStep: 'Compiling comprehensive report...',
        totalSteps: 4,
        stepNumber: 4,
        subQuestions,
        findings: allFindings,
    });

    const answer = await synthesizeReport(query, allFindings);

    onProgress({
        stage: 'complete',
        currentStep: 'Research complete!',
        totalSteps: 4,
        stepNumber: 4,
        subQuestions,
        findings: allFindings,
        intermediateAnswer: answer,
    });

    return {
        answer,
        sources: allSources,
        subQuestions,
        findings: allFindings,
        iterations,
    };
}
