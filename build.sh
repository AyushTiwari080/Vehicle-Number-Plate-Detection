#!/bin/bash
# Build script for Render deployment

# Disable Poetry and use pip directly
export POETRY_VERSION=

echo "Installing dependencies with Python 3.11..."
pip install -r requirements.txt

# Create cache directory for EasyOCR models
mkdir -p ~/.EasyOCR

echo "Build complete!"

