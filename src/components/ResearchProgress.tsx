import React from 'react';
import { ResearchProgress as ResearchProgressType } from '../services/SearxClient';

interface ResearchProgressProps {
    progress: ResearchProgressType;
}

const stageIcons: Record<string, string> = {
    thinking: '🧠',
    searching: '🔍',
    analyzing: '📊',
    synthesizing: '✍️',
    complete: '✅',
    decomposing: '🔍',
    researching: '📚',
    evaluating: '🤔',
    'follow-up': '🔄',
};

const stageLabels: Record<string, string> = {
    thinking: 'Reasoning',
    searching: 'Searching',
    analyzing: 'Analyzing Results',
    synthesizing: 'Writing Answer',
    complete: 'Complete',
    decomposing: 'Breaking Down Question',
    researching: 'Researching',
    evaluating: 'Evaluating Findings',
    'follow-up': 'Following Up',
};

const ResearchProgress: React.FC<ResearchProgressProps> = ({ progress }) => {
    return (
        <div className="research-progress">
            <div className="progress-header">
                <span className="progress-icon">{stageIcons[progress.stage] || '⏳'}</span>
                <span className="progress-stage">{stageLabels[progress.stage] || progress.stage}</span>
            </div>

            <div className="progress-bar-container">
                <div className="progress-bar progress-bar-animated" />
            </div>

            <p className="progress-step">{progress.currentStep}</p>

            {/* Show thought process for LangChain agent */}
            {progress.thought && (
                <div className="progress-thought">
                    <span className="thought-label">💭 Thought:</span>
                    <span className="thought-content">{progress.thought}</span>
                </div>
            )}

            {/* Show action being taken */}
            {progress.action && (
                <div className="progress-action">
                    <span className="action-label">⚡ Action:</span>
                    <span className="action-content">{progress.action}</span>
                </div>
            )}

            {/* Show sub-questions if available */}
            {progress.subQuestions && progress.subQuestions.length > 0 && (
                <div className="progress-subquestions">
                    <h4>Research Areas:</h4>
                    <ul>
                        {progress.subQuestions.map((q, i) => (
                            <li key={i} className={progress.findings && progress.findings.some(f => f.subQuestion === q) ? 'completed' : ''}>
                                {q}
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
};

export default ResearchProgress;
