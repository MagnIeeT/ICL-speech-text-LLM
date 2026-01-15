#!/bin/bash

# Navigate to the ICL directory
cd /home/neeraja/code/ICL-speech-text-LLM

# Get current date for commit message
DATE=$(date '+%Y-%m-%d %H:%M:%S')

# Add all changes (including deletions)
git add -A

# Only commit if there are changes
if [ -n "$(git status --porcelain)" ]; then
    git commit -m "Auto-commit: ${DATE}"
    git push origin main
    echo "Changes committed and pushed on ${DATE}"
else
    echo "No changes to commit on ${DATE}"
fi 