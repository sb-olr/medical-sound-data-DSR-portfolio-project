### Data visualization

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Audio Analysis
import glob
import IPython
import librosa
import librosa.display
import tensorflow as tf
from tensorflow import keras

#path
import os

### Chunk2

# import meta data
# Meta data csv contain different additional information about each case.
# One column contains the path to the .wav files of each case
df_meta = pd.read_csv('./CoronaHack-Respiratory-Sound-Dataset/Corona-Hack-Respiratory-Sound-Metadata.csv')


### chunk3
#Get the label (healthy / COVID) 

#split COVID STATUS column to get labels in column 'split'
df_meta['split'] = df_meta['COVID_STATUS'].str.split('_').str.get(0)
#Check for NA
df_meta.loc[:,'counting-normal'].isna().sum()
df_meta.loc[:,'split'].value_counts()

#Generate a dict to re-categorize the split column
cat_dict = {'healthy':0,'no':0,'resp':0,'recovered':0,'positive':1}

#map cat_dict to split column 
df_meta.loc[:,'split'] =  df_meta.loc[:,'split'].map(cat_dict)
df_meta2 = df_meta.dropna(subset=['split'])
df_meta2.loc[:,'split'] = df_meta2.loc[:,'split'].astype('int32')


#Extract positive USER ID
df_meta_positives = df_meta[df_meta['split'] == 1]
df_meta_negatives = df_meta[df_meta['split'] == 0]

positives = list(df_meta_positives['USER_ID'])
negatives = list(df_meta_negatives['USER_ID'])

### chunk 4
# Write function for import and preprocessing of all 9 .wav files per case (code adapted from Tristan classes) 

import cv2
def preprocess_other(sample):
  image_target_height, image_target_width = 64, 64 #setting up the shape of sample
  audio_binary = tf.io.read_file(sample) #read-in the sample as tensor
  audio, rate = tf.audio.decode_wav(audio_binary, desired_channels=1) #getting the audio and rate
  #label = sample['label']

  def py_preprocess_audio(audio):
      audio = audio.numpy().astype('float32') #set audio file as float
      #generate the mel spectrogram
      spectrogram = librosa.feature.melspectrogram(
        y=audio, n_fft=1024,  n_mels=64, hop_length=64, sr=8000, fmax=2000 #n_fft = window size, n_mels = frequency bins, hop_lenghth =jump to the right , sr = sound rate, fmax = 
      ) 

      spectrogram /= np.max(spectrogram) #devide by np.max(audio)
      spectrogram = cv2.resize(spectrogram, dsize=(image_target_height, image_target_width)) #resize the spectrogram
      spectrogram = np.expand_dims(spectrogram, axis=-1) #expand the dimension ? -Why ?
      return spectrogram

  spectrogram = tf.py_function(py_preprocess_audio, [audio], tf.float32) #apply py_process_audio function 
  spectrogram.set_shape((image_target_height, image_target_width, 1)) #set shape, include channel dimension

  return spectrogram#, label

### Chunk 5

# Create function to load and prepare data for input 
# here we want to use the 9 recordings as separate features but grouped per case as input to the auto-encoder 

#names of 9 recordings per each case (extracted from the csv meta data file from )
names_input = ['counting-normal','counting-fast','breathing-deep','breathing-shallow','cough-heavy','cough-shallow','vowel-a','vowel-e','vowel-o']
#label column from the meta data csv (#Chunk 3)
name_label = 'split'

def create_input_label(df=df_meta2,names=names_input,name_label=name_label):
    input_dic = {} #Use a dictionnary to put in the 9 records per case
    for index,name in enumerate(names):
        #print(index,name)
        path_list = df[name].tolist()
        #print(path_list[:10])
        path_name = ['./CoronaHack-Respiratory-Sound-Dataset'  + str(dir_name for dir_name in path_list)]
        sound_paths_tensor = tf.convert_to_tensor(path_name, dtype=tf.string) #convert to tensor
        sound = tf.data.Dataset.from_tensor_slices(sound_paths_tensor)
        input_dic['x_{}'.format(index)] = sound.map(lambda sample: preprocess_other(sample)).batch(32) #generating the names of recordings(features x_0 till x_8) in batch mode


    path_label = df[name_label]
    #print(path_label)
    y = tf.convert_to_tensor(path_label, dtype=tf.int16)

    return input_dic,y

### Chunk 6
x,y = create_input_label()

### Chunk 7
from tensorflow.keras import models, layers

class AutoEncoder(tf.keras.Model):
    
    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim

        # Encoder
        self.encoder_reshape = layers.Reshape((64,64,1)) #Shape as 64,64,1
        self.encoder_fc1 = layers.Dense(256, activation="relu")
        self.encoder_fc2 = layers.Dense(latent_dim, activation="relu")

        # Decoder
        self.decoder_fc1 = layers.Dense(256, activation='relu')
        self.decoder_fc2 = layers.Dense(1, activation='sigmoid')
        self.decoder_reshape = layers.Reshape((64,64,1))

        self._build_graph()

    def _build_graph(self):
        input_shape = (64,64,1)
        self.build((None,)+ input_shape)
        inputs = tf.keras.Input(shape=input_shape)
        _= self.call(inputs)

    def call(self, x):
        z = self.encode(x)
        x_new = self.decode(z)
        return x_new

    def encode(self, x):
        x = self.encoder_reshape(x)
        x = self.encoder_fc1(x)
        z = self.encoder_fc2(x)
        return z
   

    def decode(self, z):
        z = self.decoder_fc1(z)
        z = self.decoder_fc2(z)
        x = self.decoder_reshape(z)
        return x

autoencoder = AutoEncoder(32)
autoencoder.summary()

autoencoder.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy'
)

### Chunk 8

history_list = {}

history = autoencoder.fit(
    x,
    epochs = 20,
    batch_size=32

)

history_list['base'] = history