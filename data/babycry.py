import os
import shutil
import random
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import utils.utility as utility

from tensorflow import keras

random_number = random.randint(0, 100000)


def mix_up(x1, y1, x2, y2, alpha=0.2):
    alpha = np.random.random() * alpha
    alpha = np.random.beta(alpha, alpha)
    x = alpha * x2 + (1 - alpha) * x1
    y = alpha * y2 + (1 - alpha) * y1
    return x, y


def tmp_save_audio(audio, path, sr=16000):
    os.makedirs(
        os.path.join(os.getcwd(), f"batch_audio_{random_number}"), exist_ok=True
    )
    path = os.path.join(
        os.getcwd(),
        f"batch_audio_{random_number}",
        path.split("/")[-1].split(".")[0] + ".wav",
    )
    sf.write(path, audio, sr)
    return path


def augment_data(speech_path, irfile_path=None, mic_ir_file_path=None):
    speech, fs_s = sf.read(speech_path, dtype="float64")
    if fs_s != 16000:
        speech = librosa.resample(speech, orig_sr=fs_s, target_sr=16000)
        fs_s = 16000
    speech_length = speech.shape[0]
    speech = (speech - np.min(speech)) / (np.max(speech) - np.min(speech))  # normalize
    # noise = np.random.normal(0, 0.01, speech_length)
    # speech = speech + noise  # add noise

    # convolution
    if irfile_path != None:
        IR, fs_i = sf.read(irfile_path)

        if fs_i != 16000:
            IR = librosa.resample(IR, orig_sr=fs_i, target_sr=16000)
            fs_i = 16000

        temp = utility.smart_convolve(speech, IR)

        speech = np.array(temp)

    if mic_ir_file_path != None:
        mic_IR, fs_i = sf.read(mic_ir_file_path)

        if fs_i != 16000:
            mic_IR = librosa.resample(mic_IR, orig_sr=fs_i, target_sr=16000)
            fs_i = 16000

        temp = utility.smart_convolve(speech, mic_IR)

        speech = np.array(temp)

    maxval = np.max(np.fabs(speech))
    if maxval == 0:
        print("file {} not saved due to zero strength".format(speech_path))
        return -1
    if maxval >= 1:
        amp_ratio = 0.99 / maxval
        speech = speech * amp_ratio

    return speech


def augment_audio(audio_path, rir_dir, mic_dir, augment):
    if np.random.random() >= augment:
        return librosa.load(audio_path, sr=16000)[0]

    rir_path = (
        os.path.join(rir_dir, random.choice(os.listdir(rir_dir)))
        if np.random.random() >= 0.1
        else None
    )
    mic_path = (
        os.path.join(mic_dir, random.choice(os.listdir(mic_dir)))
        if np.random.random() >= 0.1
        else None
    )

    return augment_data(audio_path, rir_path, mic_path)


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
        augment=0,
        mix_up=0,
        mix_up_alpha=0.2,
        rir_dir=None,
        mic_dir=None,
        save_audio=False,
        label_smoothing=False,
    ):
        self.dir = dir
        self.spec = spec
        self.batch_size = batch_size

        self.spec_extraction = spec_extraction
        self.options = options
        self.augment = augment
        self.rir_dir = rir_dir
        self.mic_dir = mic_dir
        self.mix_up = mix_up
        self.mix_up_alpha = mix_up_alpha
        self.label_smoothing = label_smoothing

        if split != "train_loso":
            self.df = pd.read_csv(os.path.join(dir, split + ".csv"))
            print(f"Length of {split}: {len(self.df.index)}")

            if speakers != None:
                self.df = self.df[self.df["id"].isin(speakers)]

            print(f"Length of {split} after filtering: {len(self.df.index)}")
        else:
            train = pd.read_csv(os.path.join(dir, "train.csv"))
            print(f"Length of train: {len(train.index)}")
            val = pd.read_csv(os.path.join(dir, "val.csv"))
            print(f"Length of val: {len(val.index)}")
            self.df = pd.concat(
                [
                    train,
                    val,
                ]
            )
            self.df = self.df[self.df["id"].isin(speakers)]
            print(f"Length of train + val: {len(self.df.index)}")

        self.df_len = len(self.df.index)
        self.indexes = np.arange(self.df_len)
        self.input_shape = input_shape
        self.save_audio = save_audio
        if not os.path.exists(self.save_audio):
            os.makedirs(self.save_audio, exist_ok=True)

    def __len__(self):
        return int(np.floor(self.df_len / self.batch_size))

    def get_weights(self):
        j = len(self.df[self.df["label"] == "J"])
        g = len(self.df[self.df["label"] == "G"])

        return [g / (j + g), j / (j + g)]

    def __getitem__(self, index):
        batch = self.df.iloc[index * self.batch_size : (index + 1) * self.batch_size]

        audio_batch = [
            augment_audio(path, self.rir_dir, self.mic_dir, self.augment)
            for path in batch["path"]
        ]

        max_len = max(len(row) for row in audio_batch)
        audio_batch = np.array(
            [np.pad(row, (0, max_len - len(row))) for row in audio_batch]
        )

        label_batch = np.asarray(
            [[1.0, 0.0] if label == "J" else [0.0, 1.0] for label in batch["label"]]
        )

        if np.random.random() <= self.mix_up:
            # batch2 = self.df.sample(self.batch_size)
            g_indices = self.df[self.df["label"] == "G"].index
            j_indices = self.df[self.df["label"] == "J"].index
            indicies = []

            for i in range(self.batch_size):
                if label_batch[i][0] == 1.0:
                    indicies.append(random.choice(g_indices))
                else:
                    indicies.append(random.choice(j_indices))
            batch2 = self.df.iloc[indicies]

            audio_batch2 = [
                augment_audio(path, self.rir_dir, self.mic_dir, self.augment)
                for path in batch2["path"]
            ]
            max_len2 = max(len(row) for row in audio_batch2)
            if max_len2 > max_len:
                max_len = max_len2
                audio_batch = np.array(
                    [np.pad(row, (0, max_len - len(row))) for row in audio_batch]
                )
            audio_batch2 = np.array(
                [np.pad(row, (0, max_len - len(row))) for row in audio_batch2]
            )
            label_batch2 = np.asarray(
                [
                    [1.0, 0.0] if label == "J" else [0.0, 1.0]
                    for label in batch2["label"]
                ]
            )

            audio_batch, label_smooth = mix_up(
                audio_batch, label_batch, audio_batch2, label_batch2, self.mix_up_alpha
            )

            if self.label_smoothing:
                label_batch = label_smooth
            return audio_batch, label_batch

        if type(self.save_audio) == str:
            org_audio = [augment_audio(path, None, None, 0.0) for path in batch["path"]]

            sf.write(
                os.path.join(
                    self.save_audio,
                    str(len([i for i in os.listdir(self.save_audio) if not "org" in i]))
                    + ".wav",
                ),
                audio_batch[0],
                16000,
            )
            sf.write(
                os.path.join(
                    self.save_audio,
                    str(len([i for i in os.listdir(self.save_audio) if "org" in i]))
                    + "_org.wav",
                ),
                org_audio[0],
                16000,
            )
            if len(os.listdir(self.save_audio)) == 20:
                self.save_audio = False

        if self.batch_size == 1:
            audio_batch = audio_batch[0]

        return audio_batch, label_batch

    def on_epoch_end(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
