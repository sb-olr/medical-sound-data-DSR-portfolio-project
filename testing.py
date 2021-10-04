#Data visualization
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

#Audio Analysis
import glob
import IPython
import tensorflow as tf
from tensorflow import keras
import tensorflow_io as tfio
from random import shuffle
from statistics import mean
from data_prepration import Data
from models import decode, encode, VAE

# names_input = ['counting-normal','counting-fast','breathing-deep', 'breathing-shallow','cough-heavy','cough-shallow','vowel-a','vowel-e','vowel-o']


names = ['breathing-deep', 'breathing-shallow','cough-heavy','cough-shallow']

latent_dim = 2
image_target_height = 28 
image_target_width = 28

def get_spectrogram(sample):
        audio = sample
        audio = tf.reshape(sample, [-1])
        audio = tf.cast(audio, tf.float32)  # set audio file as float
        # generate the mel spectrogram
        spectrogram = tfio.audio.spectrogram(audio, nfft=1024, window=1024, stride=64)
        spectrogram = tfio.audio.melscale(
            spectrogram,
            rate=48000,
            mels=64,
            fmin=0,
            fmax=2000,  # mels = bins, fmin,fmax = frequences
        )

        spectrogram /= tf.math.reduce_max(spectrogram)  # normalization
        spectrogram = tf.expand_dims(spectrogram, axis=-1)  # add dimension 2D -> 3D
        spectrogram = tf.image.resize(
            spectrogram, (image_target_height, image_target_height)
        )  # resize in two dimensions
        spectrogram = tf.transpose(
            spectrogram, perm=(1, 0, 2)
        )  # transpose the first two axis
        spectrogram = spectrogram[::-1, :, :]  # flip the first axis(frequency)

        return spectrogram

file_name = ( "data/Corona-Hack-Respiratory-Sound-Metadata.csv" )
base_path = "data/CoronaHack-Respiratory-Sound-Dataset"

data_obj = Data(filename=file_name)
train_df, test_df = data_obj.create_df()

train_df = train_df.iloc[:100]




def get_paths(df, name):
    paths_vector = df[name]
    paths_list = df[name].values.tolist()

    path_name = []

    # Standard approach
    print("paths_vector LENGTH", len(paths_vector))
    for dir_name in paths_list:
        if dir_name is not None:
            path_name.append(base_path + str(dir_name[0]))

    # DF approach
    

    return path_name



# print("sound_tensors LENGTH", len(test_df['sound_tensors']))

# print('sound_tensors', test_df['sound_tensors'][0])
def get_sound_tensors(sound_paths):

    sound_tensor_list = [
        tfio.audio.AudioIOTensor(sound_path).to_tensor()[:300000]
        for sound_path in sound_paths
    ]

    # print("Sound Tensor List Len", sound_tensor_list)
    sound_tensor_list = [
                    sound_tensor 
                    for sound_tensor in sound_tensor_list
                    if (np.sum(sound_tensor.numpy()) != 0)
                    # if ((sound_tensor.shape[0] == 300000) and (np.sum(sound_tensor.numpy()) != 0))
                ]

    print('spectrograms LENGTH > 0 REAL', len(sound_tensor_list))

    return sound_tensor_list

def get_samples_from_tensor(sound_tensors):
    test_samples = [get_spectrogram(sound_tensor) for sound_tensor in sound_tensors]
    test_samples = [tf.expand_dims(test_sample, axis=0) for test_sample in test_samples]

    return test_samples

def find_threshold(model, train_samples):
  reconstructions = [model.predict(x_input) for x_input in train_samples]
  # provides losses of individual instances
  reconstruction_errors = tf.keras.losses.msle(train_samples, reconstructions)
  # threshold for anomaly scores
  threshold = np.mean(reconstruction_errors.numpy()) 
  # + np.std(reconstruction_errors.numpy())

  return threshold

def get_predictions(model, test_samples, threshold):
    predictions = [model.predict(x_input) for x_input in test_samples]
    # provides losses of individual instances
    test_samples = [tf.reshape(t, [-1])  for t in test_samples]
    predictions = [tf.reshape(p, [-1])  for p in predictions]

    errors = tf.keras.losses.msle(test_samples, predictions)

    print("ERRORS. ", errors)
    print("ERRORS.shape ", errors.shape)

    anomaly_mask = pd.Series(errors) > threshold

    print("anomaly_mask. ", anomaly_mask)
    preds = anomaly_mask.map(lambda x: 1.0 if x == True else 0.0)
    return preds


encoder = encode(
    latent_dim, image_target_height, image_target_width
)
decoder = decode(latent_dim)

model = VAE(encoder, decoder)


for name in names:

    name = [name]
    weights_name = 'vae' + name[0] + '-48000_checkpoint'

    test_df['full_path'] = base_path + test_df[name]
    print("full_path LENGTH", len(test_df['full_path']))

    test_df['sound_tensors'] = test_df['full_path'].apply(lambda sound_path: tfio.audio.AudioIOTensor(sound_path).to_tensor()[:300000])
    test_df = test_df.loc[test_df['sound_tensors'].apply(lambda sound_tensors: np.sum(sound_tensors)) != 0]
    print('spectrograms LENGTH > 0', len(test_df))
    y_test = test_df['split'].tolist()

    print("GET PATHS ", name)
    test_paths = get_paths(test_df, name)
    train_paths = get_paths(train_df, name)
    

    print("GET SOUND TENSORS ", name)
    train_sound_tensors = get_sound_tensors(train_paths)
    test_sound_tensors = get_sound_tensors(test_paths)

    print("GET SAMPLES ", name)
    train_samples = get_samples_from_tensor(train_sound_tensors)
    test_samples = get_samples_from_tensor(test_sound_tensors)

    model.load_weights(weights_name)

    threshold = find_threshold(model, train_samples)
    # threshold = 0.01313
    print(f"Threshold: {threshold}")
    # Threshold: 0.01001314025746261

    predictions = get_predictions(model, test_samples, threshold)
    accuracy_score(predictions, y_test)
    print(f"Accuracy: {accuracy_score(predictions, y_test)}")

    print("PREDS", predictions, "ACTUAL:", y_test)
    print("PREDS SUM", sum(predictions), "ACTUAL:", sum(y_test))
    print("PREDS LEN", len(predictions), "ACTUAL:", len(y_test))

    new_df = pd.DataFrame()
    new_df['prediction ' + name[0]] = predictions
    new_df['y_test' + name[0]] = y_test
    new_df.to_csv("Predictions" + name[0] + " .csv")


test_df.to_csv("Predictions.csv")

# print(test_df[name].values)
# print("Sound File List Len", len(path_name))
# print("Sound File List ", path_name)

# Cut tensors longer than 300k to 300k

# print([sound_path for sound_path in path_name])


# print("Tensor list", sound_tensor_list[0])

# sound_slices_train = tf.data.Dataset.from_tensor_slices(sound_tensor_list_clean_train)


# test_df['spectrograms'] = test_df['sound_tensors'].apply(lambda sound_tensor: get_spectrogram(sound_tensor))
# test_df['spectrograms'] = test_df['spectrograms'].apply(lambda spectrogram: tf.expand_dims(spectrogram, axis=0))

# # print('spectrograms', test_df['spectrograms'][1])
# print('spectrograms LENGTH', len(test_df['spectrograms']))




# print("Test Sample   ", test_samples)



# x_train = test_df['spectrograms'].to_numpy()

# print("PREDICTION ", x_output)


# print("test_df['spectrograms'] ", train_samples )
# print("x_train TYPE ", type(train_samples) )



