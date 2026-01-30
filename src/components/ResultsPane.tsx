import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { SearchResult, ResearchFinding } from '../services/SearxClient';

interface ResultsPaneProps {
    answer: string;
    sources: SearchResult[];
    isLoading: boolean;
    error: string | null;
    subQuestions?: string[];
    findings?: ResearchFinding[];
    isDeepResearch?: boolean;
}

const ResultsPane: React.FC<ResultsPaneProps> = ({
    answer,
    sources,
    isLoading,
    error,
    subQuestions = [],
    findings = [],
    isDeepResearch = false,
}) => {
    if (error) {
        return (
            <div className="results-pane results-error">
                <div className="error-icon">⚠️</div>
                <h3>Error</h3>
                <p>{error}</p>
            </div>
        );
    }

    if (isLoading) {
        return (
            <div className="results-pane results-loading">
                <div className="loading-pulse" />
                <p>{isDeepResearch ? 'Conducting deep research...' : 'Searching and synthesizing...'}</p>
            </div>
        );
    }

    if (!answer && sources.length === 0) {
        return (
            <div className="results-pane results-empty">
                <div className="empty-icon">🔍</div>
                <p>Enter a query above to start searching</p>
                {isDeepResearch && (
                    <p className="mode-hint">Deep Research mode will explore multiple angles of your question</p>
                )}
            </div>
        );
    }

    return (
        <div className="results-pane">
            {/* Deep Research: Show research journey */}
            {isDeepResearch && findings.length > 0 && (
                <div className="research-journey">
                    <h2 className="section-title">🔬 Research Journey</h2>
                    <div className="findings-accordion">
                        {findings.map((finding, idx) => (
                            <details key={idx} className="finding-item">
                                <summary className="finding-question">
                                    <span className="finding-number">{idx + 1}</span>
                                    {finding.subQuestion}
                                </summary>
                                <div className="finding-content">
                                    <h4>Key Findings:</h4>
                                    <ul>
                                        {finding.keyPoints.map((point, i) => (
                                            <li key={i}>{point}</li>
                                        ))}
                                    </ul>
                                    {finding.sources.length > 0 && (
                                        <>
                                            <h4>Sources:</h4>
                                            <ul className="finding-sources">
                                                {finding.sources.map((source, i) => (
                                                    <li key={i}>
                                                        <a href={source.url} target="_blank" rel="noopener noreferrer">
                                                            {source.title}
                                                        </a>
                                                    </li>
                                                ))}
                                            </ul>
                                        </>
                                    )}
                                </div>
                            </details>
                        ))}
                    </div>
                </div>
            )}

            {/* Main Answer */}
            <div className="answer-section">
                <h2 className="section-title">
                    {isDeepResearch ? '📋 Comprehensive Report' : '💡 Answer'}
                </h2>
                <div className="answer-content">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{answer}</ReactMarkdown>
                </div>
            </div>

            {/* Sources */}
            {sources.length > 0 && (
                <div className="sources-section">
                    <h2 className="section-title">📚 Sources ({sources.length})</h2>
                    <ul className="sources-list">
                        {sources.slice(0, 20).map((source, index) => (
                            <li key={index} className="source-item">
                                <a href={source.url} target="_blank" rel="noopener noreferrer" className="source-link">
                                    <span className="source-number">{index + 1}</span>
                                    <div className="source-info">
                                        <span className="source-title">{source.title}</span>
                                        <span className="source-url">{new URL(source.url).hostname}</span>
                                    </div>
                                </a>
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
};

export default ResultsPane;
