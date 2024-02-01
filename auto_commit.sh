#!/bin/bash

# Navigate to your project repository
cd "/home/sayem/Desktop/Hangman"

# Explicitly add the 'Old' folder recursively
find versions -type f \( -name "*.py" -o -name "*.ipynb" \) -exec git add -f {} \;
find scr -type f \( -name "*.py" -o -name "*.ipynb" \) -exec git add -f {} \;
# find report -type f -name "*.html" -exec git add -f {} \;

# Check for changes in both working directory and staged area
if [ -n "$(git diff)" ] || [ -n "$(git diff --cached)" ]; then
    # Add all other changes not mentioned in .gitignore
    git add .

    # Commit the changes
    git commit -m "Automatic commit at $(date)"

    # Determine which branch to push based on the current active branch
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

    if [ "$CURRENT_BRANCH" = "master" ]; then
        git push git@github.com:skhan61/HangmanNN.git master
    else
        echo "You are currently on the $CURRENT_BRANCH branch. Please switch to the main branch to auto-push."
    fi
else
    echo "No changes detected."
fi