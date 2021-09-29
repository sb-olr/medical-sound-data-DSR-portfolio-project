#Data visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Audio Analysis
import librosa
import librosa.display
import tensorflow as tf
#path
import os
import tensorflow_io as tfio
from scipy.io import wavfile

#import meta data
df_meta = pd.read_csv('./CoronaHack-Respiratory-Sound-Dataset/Corona-Hack-Respiratory-Sound-Metadata.csv')

import cv2
def preprocess_other(samples):
    spectograms = []
    for sample in samples:
      image_target_height, image_target_width = 64, 64 #setting up the shape of sample
      audio_binary = tf.io.read_file(sample) #read-in the sample 
      audio, rate = tf.audio.decode_wav(audio_binary, desired_channels=1) #getting the audio and rate
      #label = sample['label']

      def py_preprocess_audio(audio):
          audio = audio.numpy().astype('float32')

          spectogram = librosa.feature.melspectrogram(
            y=audio, n_fft=1024,  n_mels=64, hop_length=64, sr=8000, fmax=2000 #n_fft = window size, n_mels = frequency bins, hop_lenghth =jump to the right , sr = sound rate, fmax = 
          ) 

          spectogram /= np.max(spectogram) #devide by np.max(audio)
          spectogram = cv2.resize(spectogram, dsize=(image_target_height, image_target_width))
          spectogram = np.expand_dims(spectogram, axis=-1)
          return spectogram
      #fs, audio = wavfile.read(sample)
      spectogram = tf.py_function(py_preprocess_audio, [audio], tf.float32)
      spectogram.set_shape((image_target_height, image_target_width, 1))
      spectograms.append(spectogram)

    return spectograms


names = ['counting-normal','counting-fast','breathing-deep','breathing-shallow','cough-heavy','cough-shallow','vowel-a','vowel-e','vowel-o']
input_dic = {}
base_path = './CoronaHack-Respiratory-Sound-Dataset' 
for index,name in enumerate(names):
    path_list = df_meta[name].tolist()
    path_name = []
    for dir_name in path_list:
      path_name.append(base_path+dir_name)
    spectograms =  preprocess_other(path_name)
    x = spectograms[0].numpy()
    plt.imshow(x[:,:, 0], cmap='inferno')
    plt.show()
    assert False
    break