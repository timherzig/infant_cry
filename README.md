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

Training using the configuration found in 'config/config_cluster.yaml' the model achieves an **F1 score of 0.8977** on previously unseen data after 20 epochs. This model was trained on non-augmented data, with a batch size of 32 and a learning rate of 0.0001. The model was trained on a single GPU (RTXA6000 48GB). Other experiments were run using augmented data which previously had led to improved performance. Augmentation methods can be found the repository https://github.com/timherzig/speech_augment, and using generated room impulse responses from a modified version of anton-jeran/FAST-RIR found here: https://github.com/timherzig/FAST-RIR.


## Nov. 2023

|Config|Test F1|
|:---:|:---:|
|`config/trill/trill1_noaug.yaml`|0.818|
|`config/trill/trill1_aug5.yaml`|0.477|
|`config/jdc/jdc_noaug_bilstm.yaml`|0.655|
|`config/jdc/jdc_noaug_bilstm_bc1.yaml`|0.639|
|`config/jdc/jdc_noaug_bilstm_tiny_bc1.yaml`|0.674|
|`config/jdc/jdc_aug2_bilstm.yaml`|0.625|
|`config/jdc/jdc_aug2_bilstm_tiny.yaml`|0.626|
|`config/jdc/jdc_aug5_bilstm.yaml`|0.670|
|`config/jdc/jdc_trainable_noaug_bilstm.yaml`|0.988|
|`config/jdc/jdc_trainable_aug5_bilstm.yaml`|0.994|
|`config/jdc/jdc_trainable_noaug_bilstm_tiny.yaml`|0.982|