import os
import librosa
import torch
import numpy as np
from moviepy.editor import *
from torch.utils.data import Dataset, DataLoader

TF = "Truthful"
DC = "Deceptive"


def VideoToAudio(inputpath: str, outputpath: str):
    video = VideoFileClip(inputpath)
    # video = video.subclip(0, 90)
    audio = video.audio
    audio.write_audiofile(outputpath)


def get_all_audio(rootDir: str, outputfolder: str):
    for root, _, files in os.walk(rootDir):
        for file in files:
            print(root)
            out_root = root.replace("teamwork", "teamwork-speech")
            if not os.path.exists(out_root):
                os.makedirs(out_root)

            file_name = os.path.join(outputfolder, file)
            print(file_name)
            outputpath = file_name.replace("teamwork", "teamwork-speech")
            outputpath = outputpath.replace("mp4", "wav")

            if not os.path.exists(outputpath):
                inputpath = os.path.join(root, file)
                print(inputpath)
                print(outputpath)
                VideoToAudio(inputpath, outputpath)


def extract_all_audios(rootDir: str, outputfolder: str, sr=22050):
    get_all_audio(rootDir, outputfolder)
    for filename in os.listdir(outputfolder):
        _, typ, idx = filename.split("_")
        idx = int(idx.split(".")[0])
        signal, sample_rate = librosa.load(f"{outputfolder}/{filename}", sr=sr)
        # MFCCs
        hop_length = 512  # in num. of samples
        n_fft = 2048  # window in num. of samples
        MFCCs = librosa.feature.mfcc(
            signal, sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=13
        )
        MFCCs = MFCCs[:, :196]
        # perform stft
        stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
        # calculate abs values on complex numbers to get magnitude
        spectrogram = np.abs(stft)
        # apply logarithm to cast amplitude to Decibels
        db = librosa.amplitude_to_db(spectrogram)
        db = db[:, :196]
        mel_freq_coeff_delta = librosa.feature.delta(MFCCs, width=7)
        mel_freq_coeff_delta_delta = librosa.feature.delta(MFCCs, width=7, order=2)
        features = np.concatenate(
            (MFCCs, mel_freq_coeff_delta, mel_freq_coeff_delta_delta, db), axis=0
        )
        print(features.shape)
        features = features.T
        cache = f"./data/TF_DC/{typ}_{idx:02d}.npy"
        print(cache)
        np.save(cache, MFCCs)


class AudioDataset(Dataset):
    def __init__(self, train: bool, device=None):
        self.device = device
        self.root_path = f"../data/TF_DC/"
        self.data = []
        for filename in os.listdir(self.root_path):
            category, idx = filename.split("_")
            idx = int(idx.split(".")[0])
            if train ^ (idx <= 8):
                self.data.append(
                    (os.path.join(self.root_path, filename), category, idx)
                )
        self.mapping = {"lie": 1, "truth": 0}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file_path, category, idx = self.data[index]
        feature = np.load(file_path)
        label = np.zeros((2,))
        label[self.mapping[category]] = 1

        return (
            torch.from_numpy(feature).float().to(self.device),
            torch.from_numpy(label).float().to(self.device),
        )


if __name__ == "__main__":
    dataset = AudioDataset(False)
    train_dataloader = DataLoader(dataset, 1, True, num_workers=0)
    audio, lab = next(iter(train_dataloader))
    input_shape = audio.shape[-1]
    print(input_shape)
