# Infant Cry

Rework of the original Infant Cries implementation.

# Installation

    apt-get update && apt-get install -y libsndfile1 ffmpeg
    conda create -p ./venv python=3.9
    conda activate ./venv
    pip install -r requirements.txt

# Running

To run a training, call:

    python run.py --config 'path/to/config.yaml'

with the config linking to compatible data. 

# Results

Training using the configuration found in 'config/config_cluster.yaml' the model achieves an **F1 score of 0.8556** after 100 epochs.