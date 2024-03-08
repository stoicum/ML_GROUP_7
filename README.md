# Music Tagger

## Intro
Music genre classification based on audio processing and machine learning

## Training Data
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

## Features 
filename: name of the audio file.

length: The length of the samples in the audio file.

chroma_stft_mean & chroma_stft_var: The mean and variance of the spectrogram of tones (STFT). Tones are associated with 12 different notes, and these features capture the intensity and variation of different notes in the music.

rms_mean & rms_var: The mean and variance of the root mean square (RMS) values. This is a measure of the amplitude of the audio signal and can be used to estimate the loudness of a track.

spectral_centroid_mean & spectral_centroid_var: The mean and variance of the spectral center of mass. The spectral center of mass is a measure of the "brightness" of an audio signal, the higher the center of mass, the higher the perceived pitch.

spectral_bandwidth_mean & spectral_bandwidth_var: the mean and variance of the spectral bandwidth. This reflects the width of the spectral distribution of the audio signal, which is related to the richness of the sound quality and the fullness of the sound.

rolloff_mean & rolloff_var: The mean and variance of the spectrum rolloff points. This is typically used to estimate the spectral shape of an audio signal, indicating the proportion of low frequency (below the roll-off point) and high frequency content.

zero_crossing_rate_mean & zero_crossing_rate_var: The mean and variance of the zero crossing rate. This is the rate at which the signal changes sign and is often used to estimate the frequency content of an audio signal.

harmony_mean & harmony_var: Mean and variance of the harmony component. In a musical signal, harmony is the tonal component after percussion and noise have been removed.

perceptr_mean & perceptr_var: Mean and variance of the percussion component. This is related to tempo and beat and is usually separated from the harmonic component.

tempo: The tempo of the track, i.e. the number of beats per minute, which is the speed or rhythm of the music.

mfcc[1-20]_mean & mfcc[1-20]_var: The mean and variance of the Mel Frequency Cepstrum Coefficients (MFCCs). These are representations of sound signals and are commonly used in sound recognition and music signal processing. Each MFCC captures a different "shape" of the audio signal and can be thought of as a "fingerprint" of the audio.

