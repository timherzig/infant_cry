import os
import librosa
import numpy as np
import soundfile as sf

import utils.utility as utility

# These need to be changed for equivalent paths
RIR_PATH = "/netscratch/herzig/shared_projects/FAST-RIR/code_new/Generated_RIRs"
MIC_PATH = "/netscratch/herzig/datasets/mic_ir"
SAVE_PATH = "/netscratch/herzig/shared_projects/infant_cry/ir_examples/"
BC_EXAMPLE = "/netscratch/herzig/datasets/BabyCry_no_augment/clips/G2500452.wav"


def gen_rir_example(audio_path, rir_path, sr=16000):
    audio, a_sr = sf.read(audio_path, dtype="float64")
    rir, r_sr = sf.read(rir_path, dtype="float64")

    if a_sr != sr:
        audio = librosa.resample(audio, a_sr, sr)
    if r_sr != sr:
        rir = librosa.resample(rir, r_sr, sr)

    speech = np.array(utility.smart_convolve(audio, rir))

    maxval = np.max(np.fabs(speech))
    if maxval == 0:
        print("file not saved due to zero strength")
        return -1
    if maxval >= 1:
        amp_ratio = 0.99 / maxval
        speech = speech * amp_ratio

    return speech


def gen_examples():
    sample_rirs = os.listdir(RIR_PATH)[:10]
    sample_mics = os.listdir(MIC_PATH)[:10]

    os.makedirs(SAVE_PATH, exist_ok=True)

    b, b_sr = sf.read(BC_EXAMPLE, dtype="float64")
    if b_sr != 16000:
        b = librosa.resample(b, b_sr, 16000)
    sf.write(os.path.join(SAVE_PATH, "baby_cry.wav"), b, 16000)

    for rir in sample_rirs:
        path = os.path.join(RIR_PATH, rir)
        out = gen_rir_example(BC_EXAMPLE, path)
        sf.write(os.path.join(SAVE_PATH, rir), out, 16000)

    for mic in sample_mics:
        path = os.path.join(MIC_PATH, mic)
        out = gen_rir_example(BC_EXAMPLE, path)
        sf.write(os.path.join(SAVE_PATH, mic), out, 16000)
