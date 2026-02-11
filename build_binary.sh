#!/bin/bash
set -e

echo "📦 Installing build dependencies..."
uv sync --extra dev

echo "🏗  Building Socrates binary..."
rm -rf build dist socrates.spec

# PyInstaller command
# --onefile: Create a single executable
# --name socrates: Name the output binary 'socrates'
# --clean: Clean cache
# --hidden-import: Ensure dynamic imports are included
# --add-data: Include necessary source files if needed (though --onefile usually handles imports)

uv run pyinstaller --onefile --name socrates --clean \
    --paths "$(pwd)" \
    --hidden-import textual \
    --hidden-import textual.driver \
    --hidden-import langchain \
    --hidden-import langchain_core \
    --hidden-import langchain_openai \
    --hidden-import langgraph \
    --hidden-import tui \
    --hidden-import graph \
    --hidden-import generator \
    --hidden-import critic \
    --hidden-import tools \
    --hidden-import models \
    --hidden-import config \
    --hidden-import logger \
    --collect-all textual \
    --collect-all rich \
    cli.py

echo "✅ Build complete!"
echo "🚀 Binary location: dist/socrates"
echo "Try running: ./dist/socrates --help"
