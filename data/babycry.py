import os
import librosa
import numpy as np
import pandas as pd

from tensorflow import keras


class BabyCry(keras.utils.Sequence):
    """
    BabyCry class for loading data from the BabyCry dataset. Loads one split.
    params: dir - directory of the dataset
            split - train, val, or test
            batch_size - batch size
    """

    def __init__(
        self,
        dir,
        split,
        batch_size,
        spec,
        speakers: list = None,
        input_shape=(1, 1, 1),
        spec_extraction=None,
        options=None,
    ):
        self.dir = dir
        self.spec = spec
        self.batch_size = batch_size

        self.spec_extraction = spec_extraction
        self.options = options

        self.df = pd.read_csv(os.path.join(dir, split + ".csv"))

        if speakers != None:
            self.df = self.df[self.df["speaker"].isin(speakers)]

        self.df_len = len(self.df.index)
        self.indexes = np.arange(self.df_len)
        self.input_shape = input_shape

    def __len__(self):
        return int(np.floor(self.df_len / self.batch_size))

    def __getitem__(self, index):
        batch = self.df.iloc[index * self.batch_size : (index + 1) * self.batch_size]

        if self.spec_extraction != None:
            audio_batch = [
                np.asarray(self.spec_extraction(path, self.options.input_size)[0])
                for path in batch["path"]
            ]
            longest = 0
            for audio in audio_batch:
                if audio.shape[0] > longest:
                    longest = audio.shape[0]

            # pad all audio to the same length
            audio_batch = np.array(
                [
                    np.pad(
                        audio, ((0, longest - audio.shape[0]), (0, 0), (0, 0), (0, 0))
                    )
                    for audio in audio_batch
                ]
            )

        else:
            audio_batch = [librosa.load(path, sr=16000)[0] for path in batch["path"]]

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

        label_batch = np.asarray(
            [[1.0, 0.0] if label == "J" else [0.0, 1.0] for label in batch["label"]]
        )

        # print(f"audio_batch shape: {audio_batch.shape}")
        # print(f"label_batch shape: {label_batch.shape}")
        if self.batch_size == 1:
            audio_batch = audio_batch[0]

        return audio_batch, label_batch
