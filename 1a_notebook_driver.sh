#!/bin/bash

# Define the path to your notebook
NOTEBOOK_PATH="1_a_dataset_generation.ipynb"

# Base directory without the sample size
BASE_DIR="/media/sayem/510B93E12554BBD1/dataset/"

# List of different values for NUM_STRATIFIED_SAMPLES
SAMPLE_SIZES=(2000)

# Iterate over the sample sizes and execute the notebook for each
for SIZE in "${SAMPLE_SIZES[@]}"; do
    echo "Executing notebook with NUM_STRATIFIED_SAMPLES = $SIZE"

    # Create the full directory path for the current sample size
    BASE_DATASET_DIR="${BASE_DIR}${SIZE}"
    
    echo "base_dataset_dir: $BASE_DATASET_DIR"

    # Execute the notebook with Papermill
    papermill $NOTEBOOK_PATH - -p NUM_STRATIFIED_SAMPLES 2000 -p base_dataset_dir $BASE_DATASET_DIR

    if [ $? -eq 0 ]; then
        echo "Notebook executed successfully."
    else
        echo "Error executing the notebook for size $SIZE"
    fi
done
