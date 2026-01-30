import React, { useState, KeyboardEvent } from 'react';

interface SearchBarProps {
    onSearch: (query: string) => void;
    isLoading: boolean;
}

const SearchBar: React.FC<SearchBarProps> = ({ onSearch, isLoading }) => {
    const [query, setQuery] = useState('');

    const handleSubmit = () => {
        if (query.trim() && !isLoading) {
            onSearch(query.trim());
        }
    };

    const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter') {
            handleSubmit();
        }
    };

    return (
        <div className="search-bar">
            <input
                type="text"
                className="search-input"
                placeholder="Ask anything..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={isLoading}
            />
            <button
                className="search-button"
                onClick={handleSubmit}
                disabled={isLoading || !query.trim()}
            >
                {isLoading ? (
                    <span className="loading-spinner" />
                ) : (
                    <svg className="search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <circle cx="11" cy="11" r="8" />
                        <path d="M21 21l-4.35-4.35" />
                    </svg>
                )}
            </button>
        </div>
    );
};

export default SearchBar;
