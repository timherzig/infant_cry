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

Training using the configuration found in 'config/config_cluster.yaml' the model achieves an **F1 score of 0.8977** on previously unseen data after 20 epochs. This model was trained on non-augmented data, with a batch size of 32 and a learning rate of 0.0001. The model was trained on a single GPU (RTXA6000 48GB), and can be found in checkpoints/trill1_22. Other experiments were run using augmented data which previously had led to improved performance. Augmentation methods can be found the repository https://github.com/timherzig/speech_augment.