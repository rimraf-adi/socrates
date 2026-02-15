#!/bin/bash
set -e

# Define installation paths
INSTALL_DIR="$HOME/.local/bin"
APP_DIR="$HOME/.socrates"

echo "📦 Installing Socrates..."

# Ensure directories exist
mkdir -p "$INSTALL_DIR"
mkdir -p "$APP_DIR"

# Build the binary first
echo "🏗  Building binary..."
./build_binary.sh

# Copy binary to install location
echo "📂 Copying binary to $INSTALL_DIR..."
cp dist/socrates "$INSTALL_DIR/socrates"
chmod +x "$INSTALL_DIR/socrates"

# Check if INSTALL_DIR is in PATH
if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
    echo "⚠️  $INSTALL_DIR is not in your PATH."
    
    SHELL_NAME=$(basename "$SHELL")
    RC_FILE=""
    
    if [ "$SHELL_NAME" = "zsh" ]; then
        RC_FILE="$HOME/.zshrc"
    elif [ "$SHELL_NAME" = "bash" ]; then
        RC_FILE="$HOME/.bashrc"
    fi
    
    if [ -n "$RC_FILE" ]; then
        echo "🔧 Adding $INSTALL_DIR to $RC_FILE..."
        echo "" >> "$RC_FILE"
        echo "# Socrates Agent" >> "$RC_FILE"
        echo "export PATH=\"\$PATH:$INSTALL_DIR\"" >> "$RC_FILE"
        echo "✅ Added to PATH. Please restart your terminal or run: source $RC_FILE"
    else
        echo "❌ Could not determine shell config file. Please add this manually:"
        echo "   export PATH=\"\$PATH:$INSTALL_DIR\""
    fi
else
    echo "✅ $INSTALL_DIR is already in PATH."
fi

echo "🎉 Installation complete!"
echo "Run 'socrates --help' to get started."
