# Neural Network Training Utility

This repository contains a utility script for training neural networks using TensorFlow. The script takes a training dataset CSV file as input, preprocesses the data, trains a neural network model, and saves both the trained model and a classification report.

## Requirements

- Python 3.8
- Conda

## Installation

### Step 1: Create and Activate Conda Environment

1. **Create a Conda Environment**:
    ```sh
    conda create --name neuralnet-env python=3.8
    ```

2. **Activate the Conda Environment**:
    ```sh
    conda activate neuralnet-env
    ```

### Step 2: Install Required Packages

You can install all the required packages using `conda` and `pip`. Run the following commands:

```sh
# Install core packages using conda
conda install numpy pandas scikit-learn

# Install TensorFlow
conda install -c conda-forge tensorflow

# Install tqdm for progress tracking (if you plan to use it in other scripts)
conda install -c conda-forge tqdm

# Optional: If you prefer to install TensorFlow via pip for the latest version
# pip install tensorflow

Alternatively, you can create the environment from the provided YAML file:

1. **Create a file named `environment.yml` with the following content**:

    ```yaml
    name: neuralnet-env
    channels:
      - defaults
      - conda-forge
    dependencies:
      - python=3.8
      - numpy
      - pandas
      - scikit-learn
      - tensorflow
      - pip
      - pip:
          - tqdm
    ```

2. **Create the environment using the YAML file**:

    ```sh
    conda env create -f environment.yml
    ```

3. **Activate the Environment**:

    ```sh
    conda activate neuralnet-env
    ```

## Usage

### Command-Line Arguments

- `training_file`: Path to the training CSV file.
- `--columns_to_exclude`: List of columns to exclude from training (optional).
- `--target_column`: Name of the target column.

### Example Command

```sh
python train_model.py path/to/training_data.csv --columns_to_exclude column1 column2 --target_column target
