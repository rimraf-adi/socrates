// LangChain-based Agentic Deep Research Engine
// Uses ReAct pattern with custom tools for SearXNG search

import { ChatGroq } from '@langchain/groq';
import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';

const SEARXNG_BASE_URL = process.env.VITE_SEARXNG_BASE_URL || 'http://localhost:8080';
const GROQ_API_KEY = process.env.VITE_GROQ_API_KEY || '';

export interface SearchResult {
    title: string;
    url: string;
    snippet: string;
}

export interface ResearchProgress {
    stage: 'thinking' | 'searching' | 'analyzing' | 'synthesizing' | 'complete';
    currentStep: string;
    thought?: string;
    action?: string;
    observation?: string;
}

export interface DeepResearchResult {
    answer: string;
    sources: SearchResult[];
    thoughtProcess: string[];
    iterations: number;
}

// Create the Groq LLM instance
function createLLM() {
    return new ChatGroq({
        apiKey: GROQ_API_KEY,
        model: 'llama-3.3-70b-versatile',
        temperature: 0.3,
    });
}

// SearXNG Search Tool
function createSearchTool(allSources: SearchResult[]) {
    return new DynamicStructuredTool({
        name: 'web_search',
        description: 'Search the web for information using SearXNG. Use this to find current information about any topic. Returns titles, URLs, and snippets from search results.',
        schema: z.object({
            query: z.string().describe('The search query to look up'),
        }),
        func: async ({ query }) => {
            try {
                const params = new URLSearchParams({
                    q: query,
                    format: 'json',
                    categories: 'general',
                });

                const response = await fetch(`${SEARXNG_BASE_URL}/search?${params.toString()}`);

                if (!response.ok) {
                    return `Search failed with status ${response.status}`;
                }

                const data = await response.json();
                const results: SearchResult[] = data.results.slice(0, 5).map((result: any) => ({
                    title: result.title || 'Untitled',
                    url: result.url,
                    snippet: result.content || '',
                }));

                // Store sources for later
                allSources.push(...results);

                // Format for agent
                return results.map((r, i) =>
                    `[${i + 1}] ${r.title}\nURL: ${r.url}\nSnippet: ${r.snippet}`
                ).join('\n\n');
            } catch (error) {
                return `Search error: ${error instanceof Error ? error.message : 'Unknown error'}`;
            }
        },
    });
}

// Create the research agent
export async function conductLangChainResearch(
    query: string,
    onProgress: (progress: ResearchProgress) => void
): Promise<DeepResearchResult> {
    const allSources: SearchResult[] = [];
    const thoughtProcess: string[] = [];
    let iterations = 0;

    onProgress({
        stage: 'thinking',
        currentStep: 'Initializing research agent...',
    });

    const llm = createLLM();
    const searchTool = createSearchTool(allSources);

    // Use a custom ReAct-style prompt
    const systemPrompt = `You are Socrates, an expert research assistant. Your goal is to provide comprehensive, well-researched answers.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now have enough information to provide a comprehensive answer
Final Answer: the final answer to the original input question

Important guidelines:
1. Always search for multiple aspects of a question
2. Cross-reference information from different sources
3. Cite sources using [1], [2], etc. in your final answer
4. If initial search doesn't provide enough info, search with different queries
5. Structure your final answer with clear sections using markdown

Begin!

Question: {input}
Thought:{agent_scratchpad}`;

    try {
        // Create a simple agent executor manually since we're using a custom prompt
        const tools = [searchTool];

        // Use the agent executor with streaming for progress updates
        const maxIterations = 8;
        let currentInput = query;
        let scratchpad = '';
        let finalAnswer = '';

        while (iterations < maxIterations) {
            iterations++;

            const prompt = systemPrompt
                .replace('{tools}', tools.map(t => `${t.name}: ${t.description}`).join('\n'))
                .replace('{tool_names}', tools.map(t => t.name).join(', '))
                .replace('{input}', currentInput)
                .replace('{agent_scratchpad}', scratchpad);

            onProgress({
                stage: 'thinking',
                currentStep: `Thinking... (iteration ${iterations})`,
                thought: `Analyzing how to answer: "${query.substring(0, 50)}..."`,
            });

            const response = await llm.invoke(prompt);
            const content = typeof response.content === 'string' ? response.content : '';

            thoughtProcess.push(`Iteration ${iterations}: ${content.substring(0, 200)}...`);

            // Parse the response for Action or Final Answer
            const finalAnswerMatch = content.match(/Final Answer:\s*([\s\S]*?)$/i);
            if (finalAnswerMatch) {
                finalAnswer = finalAnswerMatch[1].trim();
                break;
            }

            const actionMatch = content.match(/Action:\s*(\w+)/i);
            const actionInputMatch = content.match(/Action Input:\s*(.+?)(?=\n|$)/i);

            if (actionMatch && actionInputMatch) {
                const action = actionMatch[1];
                const actionInput = actionInputMatch[1].trim();

                onProgress({
                    stage: 'searching',
                    currentStep: `Searching: "${actionInput.substring(0, 40)}..."`,
                    action: action,
                });

                // Execute the tool
                let observation = '';
                if (action.toLowerCase() === 'web_search') {
                    observation = await searchTool.invoke({ query: actionInput });
                } else {
                    observation = `Unknown tool: ${action}`;
                }

                onProgress({
                    stage: 'analyzing',
                    currentStep: `Analyzing ${allSources.length} sources...`,
                    observation: observation.substring(0, 100) + '...',
                });

                // Add to scratchpad
                scratchpad += `\nThought: ${content.match(/Thought:\s*(.+?)(?=Action:|$)/is)?.[1]?.trim() || ''}`;
                scratchpad += `\nAction: ${action}`;
                scratchpad += `\nAction Input: ${actionInput}`;
                scratchpad += `\nObservation: ${observation}`;
            } else {
                // No clear action, assume it's a final answer attempt
                finalAnswer = content;
                break;
            }
        }

        onProgress({
            stage: 'synthesizing',
            currentStep: 'Compiling final answer...',
        });

        // If we hit max iterations without a final answer, synthesize one
        if (!finalAnswer && allSources.length > 0) {
            const context = allSources.map((s, i) =>
                `[${i + 1}] ${s.title}: ${s.snippet}`
            ).join('\n\n');

            const synthesisResponse = await llm.invoke(`
Based on your research, provide a comprehensive answer to: "${query}"

Research findings:
${context}

Format your answer with:
1. Brief executive summary
2. Main findings with source citations [1], [2], etc.
3. Key takeaways

Use markdown formatting.`);

            finalAnswer = typeof synthesisResponse.content === 'string'
                ? synthesisResponse.content
                : 'Unable to generate answer.';
        }

        onProgress({
            stage: 'complete',
            currentStep: 'Research complete!',
        });

        return {
            answer: finalAnswer || 'No answer could be generated.',
            sources: allSources,
            thoughtProcess,
            iterations,
        };
    } catch (error) {
        console.error('LangChain research error:', error);
        throw error;
    }
}
