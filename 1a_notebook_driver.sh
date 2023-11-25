#!/bin/bash

# Define the path to the notebook
NOTEBOOK_PATH="/home/sayem/Desktop/Hangman/1_a_dataset_generation.ipynb"

# Run the notebook using papermill
# Note: The output path is the same as the input path
papermill $NOTEBOOK_PATH $NOTEBOOK_PATH
