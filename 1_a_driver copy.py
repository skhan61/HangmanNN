import os
import tempfile

import papermill as pm


def run_notebook_with_parameter(input_notebook, param_value):
    with tempfile.NamedTemporaryFile(suffix=".ipynb", delete=True) as temp_output:
        # Execute the notebook with the specified parameter
        pm.execute_notebook(
            input_notebook,
            temp_output.name,
            parameters={'NUM_STRATIFIED_SAMPLES': param_value}
        )

        # The output notebook is automatically deleted due to delete=True


# List of values to iterate over
num_stratified_samples_values = [75000, 100000]

# Path to your input notebook
input_notebook_path = '1_a_dataset_generation.ipynb'

# Run the notebook with each value
for value in num_stratified_samples_values:
    run_notebook_with_parameter(input_notebook_path, value)
