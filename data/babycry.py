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


def augment_data(speech_path, irfile_path):
    speech, fs_s = sf.read(speech_path, dtype="float64")
    speech_length = speech.shape[0]
    speech = (speech - np.min(speech)) / (np.max(speech) - np.min(speech))  # normalize
    noise = np.random.normal(0, 0.01, speech_length)
    speech = speech + noise  # add noise

    if speech_length > 96000:
        speech = speech[0:96000]
        # sf.write(process_full_path,IR,fs_s)
    else:
        zeros_len = 96000 - speech_length
        zeros_lis = np.zeros(zeros_len)
        speech = np.concatenate([speech, zeros_lis])

    if np.issubdtype(speech.dtype, np.integer):
        speech = utility.pcm2float(speech, "float32")
    # convolution
    if irfile_path:
        IR, fs_i = sf.read(irfile_path)

        IR_length = IR.shape[0]

        if IR_length > fs_s:
            IR = IR[0:fs_s, :]
        else:
            zeros_len = fs_s - IR_length
            zeros_lis = np.zeros([zeros_len])
            IR = np.concatenate([IR, zeros_lis])

        if np.issubdtype(IR.dtype, np.integer):
            IR = utility.pcm2float(IR, "float32")
        # temp0 = utility.smart_convolve(speech, IR[:, 0])
        # temp1 = utility.smart_convolve(speech, IR[:, 1])

        # temp = np.transpose(np.concatenate(([temp0], [temp1]), axis=0))
        temp = utility.smart_convolve(speech, IR)

        speech = np.array(temp)

    maxval = np.max(np.fabs(speech))
    if maxval == 0:
        print("file {} not saved due to zero strength".format(speech_path))
        return -1
    if maxval >= 1:
        amp_ratio = 0.99 / maxval
        speech = speech * amp_ratio

    return speech


def augment_audio(audio_path, rar_dir, augment):
    if np.random.random() >= augment:
        return librosa.load(audio_path, sr=16000)[0]

    rar_path = os.path.join(rar_dir, random.choice(os.listdir(rar_dir)))

    return augment_data(audio_path, rar_path)


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
        rir_dir=None,
        save_audio=False,
    ):
        self.dir = dir
        self.spec = spec
        self.batch_size = batch_size

        self.spec_extraction = spec_extraction
        self.options = options
        self.augment = augment
        self.rir_dir = rir_dir

        self.df = pd.read_csv(os.path.join(dir, split + ".csv"))

        if speakers != None:
            self.df = self.df[self.df["speaker"].isin(speakers)]

        self.df_len = len(self.df.index)
        self.indexes = np.arange(self.df_len)
        self.input_shape = input_shape
        self.save_audio = save_audio
        if not os.path.exists(save_audio):
            os.makedirs(save_audio, exist_ok=True)

    def __len__(self):
        return int(np.floor(self.df_len / self.batch_size))

    def __getitem__(self, index):
        batch = self.df.iloc[index * self.batch_size : (index + 1) * self.batch_size]

        if self.spec_extraction != None:
            tmp_audio_batch = [
                tmp_save_audio(augment_audio(x, self.rir_dir, self.augment), x)
                for x in batch["path"]
            ]

            if type(self.save_audio) == str:
                shutil.copy(tmp_audio_batch[0], self.save_audio)
                if len(os.listdir(self.save_audio)) == 10:
                    self.save_audio = False

            audio_batch = [
                np.asarray(self.spec_extraction(path, self.options.input_size)[0])
                for path in batch["path"]
            ]
            for x in tmp_audio_batch:
                os.remove(x)

            os.rmdir(os.path.join(os.getcwd(), f"batch_audio_{random_number}"))

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
            audio_batch = [
                augment_audio(path, self.rir_dir, self.augment)
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

        label_batch = np.asarray(
            [[1.0, 0.0] if label == "J" else [0.0, 1.0] for label in batch["label"]]
        )

        if self.batch_size == 1:
            audio_batch = audio_batch[0]

        return audio_batch, label_batch

    def on_epoch_end(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
