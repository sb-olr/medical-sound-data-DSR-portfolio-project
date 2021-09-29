from models import AutoEncoder, VAE, encode, decode

import tensorflow as tf
import tensorflow_io as tfio
from tensorflow import keras


class Train:
    def __init__(
        self, base_path, df, names, name_labels, image_target_height, image_target_width
    ):
        self.base_path = base_path
        self.df = df
        self.names = names
        self.name_label = name_labels
        self.image_target_height = image_target_height
        self.image_target_width = image_target_width

    def get_spectrogram(self, sample):
        audio = sample
        audio = tf.reshape(sample, [-1])
        audio = tf.cast(audio, tf.float32)  # set audio file as float

        # generate the mel spectrogram
        spectrogram = tfio.audio.spectrogram(audio, nfft=1024, window=1024, stride=64)

        spectrogram = tfio.audio.melscale(
            spectrogram,
            rate=8000,
            mels=64,
            fmin=0,
            fmax=2000,  # mels = bins, fmin,fmax = frequences
        )

        spectrogram /= tf.math.reduce_max(spectrogram)  # normalization
        spectrogram = tf.expand_dims(spectrogram, axis=-1)  # add dimension 2D -> 3D
        spectrogram = tf.image.resize(
            spectrogram, (self.image_target_height, self.image_target_height)
        )  # resize in two dimensions
        spectrogram = tf.transpose(
            spectrogram, perm=(1, 0, 2)
        )  # transpose the first two axis
        spectrogram = spectrogram[::-1, :, :]  # flip the first axis(frequency)

        return spectrogram

    def create_input_label(self):
        input_dic = {}  # Use a dictionnary to put in the 9 records per case
        for index, name in enumerate(self.names):
            path_list = self.df[name].tolist()
            path_name = []
            print("create path name")
            for dir_name in path_list:
                if dir_name is not None:
                    path_name.append(self.base_path + str(dir_name))
            sound_tensor_list = [
                tfio.audio.AudioIOTensor(sound_path).to_tensor()[:300000]
                for sound_path in path_name
            ]
            sound_tensor_list_clean = [
                sound_tensor
                for sound_tensor in sound_tensor_list
                if sound_tensor.shape[0] == 300000
            ]
            sound_slices = tf.data.Dataset.from_tensor_slices(sound_tensor_list_clean)
            input_dic["x_{}".format(index)] = sound_slices.map(
                lambda sample: self.get_spectrogram(sample)
            )  # generating the names of recordings(features x_0 till x_8) in batch mode

        path_label = self.df[self.name_label]
        y = tf.convert_to_tensor(path_label, dtype=tf.int16)

        return input_dic, y

    def train(self, model_name, latent_dim, learning_rate, batch_size=256, epochs=50):

        model_path = model_name + ".hdf5"
        if model_name == "autoencoder":
            model = AutoEncoder(latent_dim)
            model.summary()

            model.compile(optimizer="rmsprop", loss="binary_crossentropy")

        elif model_name == "vae":
            encoder = encode(
                latent_dim, self.image_target_height, self.image_target_width
            )
            decoder = decode()
            model = VAE(encoder, decoder)
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

        x_input, _ = self.create_input_label()
        dataset = tf.data.Dataset.zip((x_input[0], x_input[0]))

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            model_path, monitor="loss", save_best_only=True, mode="min", verbose=0
        )
        callbacks = [checkpoint]
        history = model.fit(
            dataset.batch(batch_size),
            epochs=epochs,
            callbacks=callbacks,
        )

        return history
