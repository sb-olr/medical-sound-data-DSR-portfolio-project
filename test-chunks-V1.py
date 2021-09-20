
import tensorflow_io as tfio
import matplotlib.pyplot as plt
from IPython.display import Audio, display

def wav_to_mel(file_path):
  image_target_height, image_target_width = 64, 64
  audio_binary = tf.io.read_file(file_path)
  audio, rate = tf.audio.decode_wav(audio_binary, desired_channels=1)
  #waveform = tf.squeeze(audio, axis=-1)
  audio = audio[:,0]
  print(audio.shape)
  position = tfio.audio.trim(audio, axis=0, epsilon=0.1)
  print(position[0])
  start = position[0]
  end = position[1]
  audio= audio[start:end]

  # Normalize audio data
  audio = tf.cast(audio, tf.float32) / 32768.0  # Max int for audio data
  # Create the spectogram from audio data
  spectrogram = tfio.audio.spectrogram(
      audio, nfft=1024, window=128, stride=64
  )
  # Turn spectrogram into mel spectrogram
  spectrogram = tfio.audio.melscale(
      spectrogram, rate=rate, mels=64, fmin=0, fmax=2000
  )

  #spectrogram /= np.max(audio)
  spectrogram /= tf.math.reduce_max(spectrogram) # Normalize
  spectrogram = tf.expand_dims(spectrogram, axis=-1) # 2D -> 3D
  spectrogram = tf.image.resize(spectrogram, (image_target_height, image_target_width)) # Resize the picture
  spectrogram = tf.transpose(spectrogram, perm=(1, 0, 2)) # Swap the first two axis
  spectrogram = spectrogram[::-1, :, :] # Flip the first axis (frequency)

  display(Audio(audio, rate=8000))
   
  return spectrogram
path = './CoronaHack-Respiratory-Sound-Dataset' + df_meta.iloc[16,'counting-normal']
mel = wav_to_mel(path)
print(mel.shape)
#mel_predict = tf.expand_dims(mel, axis=0)
#print(mel_predict.shape)

#prediction = model.predict(mel_predict)
#print(prediction)
#plt.plot(prediction[0])

plt.imshow(mel[::-1,:], cmap='inferno') #flipping upside down
plt.show()
plt.close()



def preprocess(sample):
    audio_binary = tf.io.read_file(sample)
    audio, rate = tf.audio.decode_wav(audio_binary, desired_channels=1)

    #audio = sample['audio']
    #label = sample['label']
    audio = tf.cast(audio, tf.float32)/ 32768.0 # into float32 and Data types max range for float32 for normalization

    #here it is 1-D sequence of amplitude numbers
    spectrogram = tfio.audio.spectrogram( 
        audio, nfft=1024, window=1024, stride=64
    )

    #here it is 2-D
    spectrogram = tfio.audio.melscale(
        spectrogram, rate=8000, mels=64, fmin=0, fmax=2000 #mels = bins, fmin,fmax = frequences
    )
    #from melscale we need to:
    spectrogram /= tf.math.reduce_max(spectrogram) #normalization
    spectrogram = tf.expand_dims(spectrogram, axis=-1) #add dimension 2D -> 3D
    spectrogram = tf.image.resize(spectrogram, (64,64)) #resize in two dimensions
    
    spectrogram = tf.transpose(spectrogram, perm=(1,0,2)) #transpose the first two axis

    spectrogram = spectrogram[::-1, :, :] #flip the first axis(frequency)
    
    return spectrogram #, label