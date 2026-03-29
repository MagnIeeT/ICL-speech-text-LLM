#!/bin/bash

DATE=$(date '+%Y-%m-%d %H:%M:%S')

cd /home/harinis/ICL_qwen_run/ICL-speech-text-LLM

git checkout harinisri

git add -A

if [ -n "$(git status --porcelain)" ]; then
    git commit -m "Auto-commit: ${DATE}"
    git push origin harinisri
    echo "✅ Changes pushed on ${DATE}"
else
    echo "⚠️ No changes to commit"
fi