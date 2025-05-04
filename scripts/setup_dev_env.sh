#!/bin/bash

set -e  # Exit immediately if any command fails

echo "🚀 Setting up strategy-lab development environment..."

# Step 1: Check if poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry not found. Please install Poetry first: curl -sSL https://install.python-poetry.org/ | python3.13 -"
    exit 1
fi

# Step 2: Install dependencies using Poetry
echo "📦 Installing project dependencies..."
poetry lock
poetry install

# Step 3: Install pre-commit hooks
echo "🔧 Installing pre-commit hooks..."
poetry run pre-commit install

# Step 4: Make detect-secrets hook script executable
if [ -f "./scripts/detect-secrets-hook" ]; then
    echo "🔒 Making detect-secrets hook script executable..."
    chmod +x ./scripts/detect-secrets-hook
else
    echo "⚠️ Warning: detect-secrets-hook script not found in ./scripts/"
fi

# Step 5: Pre-run pre-commit on all files to validate
echo "✅ Running pre-commit check on all files..."
poetry run pre-commit run --all-files || true

echo "🎉 Setup completed successfully!"
