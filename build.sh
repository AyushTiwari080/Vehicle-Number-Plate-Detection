#!/bin/bash
# Build script for Render deployment

echo "Installing dependencies..."
pip install -r requirements.txt

# Create cache directory for EasyOCR models
mkdir -p ~/.EasyOCR

echo "Build complete!"

