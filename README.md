# Infant Cry

Implementation for the detection of infant cries using the [TRILLsson](https://arxiv.org/abs/2203.00236) speech representations.

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

Training using the configuration found in 'config/config_cluster.yaml' the model achieves an **F1 score of 0.899** on previously unseen data (unseen babies). This model was trained on non-augmented data, with a batch size of 8 and a learning rate of 0.001. The model was trained on a single GPU (RTXA6000 48GB). Other experiments were run using augmented data which previously had led to improved performance. Augmentation methods can be found the repository https://github.com/timherzig/speech_augment, and using generated room impulse responses from a modified version of anton-jeran/FAST-RIR found here: https://github.com/timherzig/FAST-RIR.

