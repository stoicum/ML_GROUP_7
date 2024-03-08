import os
import traceback

import librosa
import numpy as np

import csv

audioFileFolderPath = "archive/Data/genres_original/"

audioFileFoldersPath = os.listdir(audioFileFolderPath)
print(audioFileFoldersPath)

audioFeatures = []

csvTags = ["filename", "length", "chroma_stft_mean", "chroma_stft_var", "rms_mean", "rms_var"
    , "spectral_centroid_mean", "spectral_centroid_var", "spectral_bandwidth_mean", "spectral_bandwidth_var"
    , "rolloff_mean", "rolloff_var", "zero_crossing_rate_mean", "zero_crossing_rate_var", "harmony_mean", "harmony_var"
    , "tempo"]
for i in range(20):
    csvTags.append(f'mfcc{i + 1}_mean')
    csvTags.append(f'mfcc{i + 1}_var')
csvTags.append("label")

audioFeatures.append(csvTags)

for folderName in audioFileFoldersPath:
    path = audioFileFolderPath + folderName + "/"
    audioFileNames = os.listdir(path)

    for audioFileName in audioFileNames:
        print(path + audioFileName)
        try:
            y, sr = librosa.load(path + audioFileName, duration=30)
        except Exception as e:
            print("Processing File " + audioFileName + ". Error occurred: " + str(e))
            traceback.print_stack()
            continue

        features = []
        features.append(audioFileName)  # filename

        features.append(30)  # length TODO: correct it later?

        # chroma_stft
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        features.append(np.mean(chroma_stft))
        features.append(np.var(chroma_stft))

        # RMS
        rms = librosa.feature.rms(y=y)
        features.append(np.mean(rms))
        features.append(np.var(rms))

        # spectral_centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.append(np.mean(spectral_centroids))
        features.append(np.var(spectral_centroids))

        # spectral_bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features.append(np.mean(spectral_bandwidth))
        features.append(np.var(spectral_bandwidth))

        # rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features.append(np.mean(spectral_rolloff))
        features.append(np.var(spectral_rolloff))

        # zero_crossing_rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        features.append(np.mean(zero_crossing_rate))
        features.append(np.var(zero_crossing_rate))

        # no perceptr

        # harmony
        harmony = librosa.effects.harmonic(y)
        features.append(np.mean(harmony))
        features.append(np.var(harmony))

        # tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features.append(tempo)

        # MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  # 20 mfcc

        for i in range(20):
            features.append(np.mean(mfccs[i]))
            features.append(np.var(mfccs[i]))

        features.append(folderName)
        audioFeatures.append(features)

with open('features_30_sec.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(audioFeatures)
