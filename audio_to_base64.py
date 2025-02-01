import wave
import torch
import matplotlib.pyplot as plt
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from PIL import Image
import os
import numpy as np
import base64

def audio_to_base64(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    mel_specgram = T.MelSpectrogram(sample_rate, n_fft = 1600, win_length=1600,f_min = 100, f_max = 4000)(waveform)
    mel_specgram = T.AmplitudeToDB()(mel_specgram)
    mel_specgram = mel_specgram.squeeze(0)
    mel_specgram = mel_specgram.numpy()
    #print(mel_specgram.shape)
    mel_specgram = mel_specgram[:128, :]
    # mel_specgram = mel_specgram[::-1, :]
    mel_specgram = mel_specgram - mel_specgram.min()
    mel_specgram = mel_specgram / mel_specgram.max()
    mel_specgram = np.ascontiguousarray(mel_specgram)
    base64_img = base64.b64encode(mel_specgram)
    return base64_img