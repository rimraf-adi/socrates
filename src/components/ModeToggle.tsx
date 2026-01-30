import React from 'react';
import { SearchMode } from '../App';

interface ModeToggleProps {
    mode: SearchMode;
    onModeChange: (mode: SearchMode) => void;
}

const ModeToggle: React.FC<ModeToggleProps> = ({ mode, onModeChange }) => {
    return (
        <div className="mode-toggle">
            <button
                className={`mode-button ${mode === 'simple' ? 'active' : ''}`}
                onClick={() => onModeChange('simple')}
            >
                <span className="mode-icon">⚡</span>
                <span className="mode-label">Simple Search</span>
            </button>
            <button
                className={`mode-button ${mode === 'deep' ? 'active' : ''}`}
                onClick={() => onModeChange('deep')}
            >
                <span className="mode-icon">🔬</span>
                <span className="mode-label">Deep Research</span>
            </button>
        </div>
    );
};

export default ModeToggle;
