import os
import librosa
import numpy as np
import pandas as pd

from tensorflow import keras


class BabyCry(keras.utils.sequence):
    """
    BabyCry class for loading data from the BabyCry dataset. Loads one split.
    params: dir - directory of the dataset
            split - train, val, or test
            batch_size - batch size
    """

    def __init__(self, dir, split, batch_size):
        self.dir = dir
        self.batch_size = batch_size

        self.df = pd.read_csv(os.path.join(dir, split + ".csv"))

        self.df_len = len(self.df.index)
        self.indexes = np.arange(self.df_len)

    def __len__(self):
        return int(np.floor(self.df_len / self.batch_size))

    def __getitem__(self, index):
        batch = self.df.iloc[index * self.batch_size : (index + 1) * self.batch_size]

        audio_batch = [
            librosa.load(os.path.join(self.root_dir, path), sr=16000)[0]
            for path in batch["path"]
        ]

        if self.spec:
            audio_batch = [
                librosa.feature.melspectrogram(
                    x, sr=16000, n_mels=self.input_shape[1], fmax=8000
                )[: self.input_shape[0]]
                for x in audio_batch
            ]
        else:
            max_len = max(len(row) for row in audio_batch)
            audio_batch = np.array(
                [np.pad(row, (0, max_len - len(row))) for row in audio_batch]
            )

        label_batch = np.asarray([0 if label == "J" else 1 for label in batch["label"]])

        return audio_batch, label_batch
