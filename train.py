from models import AutoEncoder, VAE, encode, decode

import tensorflow as tf
import tensorflow_io as tfio
from tensorflow import keras
import numpy as np
from random import shuffle



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
        input_dic_val = {}
        for index, name in enumerate(self.names):
            path_list = self.df[name].tolist()
            path_name = []
            
            for dir_name in path_list:
                if dir_name is not None:
                    path_name.append(self.base_path + str(dir_name))

            # Cut tensors longer than 300k to 300k
            sound_tensor_list = [
                tfio.audio.AudioIOTensor(sound_path).to_tensor()[:300000]
                for sound_path in path_name
            ]
            print(" Sound File List Len", len(path_name))
            print(" Sound Tensor List Len", len(sound_tensor_list))
            # Take only non zero and at least 300k length tensors
            sound_tensor_list_clean = [
                sound_tensor 
                for sound_tensor in sound_tensor_list
                if ((sound_tensor.shape[0] == 300000) and (np.sum(sound_tensor.numpy())>0))
            ]
            print(" ´Clean´ Sound Tensor List Len", len(sound_tensor_list_clean))
            
            # shuffle index and create train and validation
            index_shuffle = list(range(len(sound_tensor_list_clean)))

            shuffle(index_shuffle)

            sound_tensor_list_schuffle = []
            for i in index_shuffle:
                sound_tensor_list_schuffle.append(sound_tensor_list_clean[i])
            train_index =  int(0.8*len(sound_tensor_list_clean))
            sound_tensor_list_clean_train = sound_tensor_list_schuffle[:train_index]
            sound_tensor_list_clean_vali = sound_tensor_list_schuffle[train_index:]
            sound_slices_train = tf.data.Dataset.from_tensor_slices(sound_tensor_list_clean_train)
            sound_slices_vali = tf.data.Dataset.from_tensor_slices(sound_tensor_list_clean_vali)
            input_dic["x_{}".format(index)] = sound_slices_train.map(
                lambda sample: self.get_spectrogram(sample)
            )  # generating the names of recordings(features x_0 till x_8) in batch mode
            input_dic_val["x_{}".format(index)] = sound_slices_vali.map(
                lambda sample: self.get_spectrogram(sample)
            )
            
            # sound_slices = tf.data.Dataset.from_tensor_slices(sound_tensor_list_clean)
            # train_index =  int(0.8*len(sound_tensor_list_clean))
            # input_dic["x_{}".format(index)] = sound_slices[:train_index].map(
            #     lambda sample: self.get_spectrogram(sample)
            # )  # generating the names of recordings(features x_0 till x_8) in batch mode
            # input_dic_val["x_{}".format(index)] = sound_slices[train_index:].map(
            #     lambda sample: self.get_spectrogram(sample)
            # )  
        path_label = self.df[self.name_label]
        y = tf.convert_to_tensor(path_label, dtype=tf.int16)

        return input_dic,input_dic_val, y

    def train(self, model_name, learning_rate, latent_dim, batch_size=256, epochs=50):
        
        # Create inputs for model
        x_input,x_input_val, _ = self.create_input_label()

        # Setup the model and input data depending on it's type
        if model_name == "autoencoder":
            model = AutoEncoder(latent_dim,self.image_target_height,self.image_target_width)
            model.summary()

            #model.compile(optimizer="rmsprop", loss="binary_crossentropy")
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),loss="binary_crossentropy")

            dataset = tf.data.Dataset.zip((x_input['x_0'], x_input['x_0'])).batch(batch_size)
            val_dataset = tf.data.Dataset.zip((x_input_val['x_0'], x_input_val['x_0'])).batch(batch_size)

        elif model_name == "vae":
            encoder = encode(
                latent_dim, self.image_target_height, self.image_target_width
            )

            decoder = decode(latent_dim)
            model = VAE(encoder, decoder)
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
            print(x_input.keys())
            dataset = x_input['x_0'].batch(batch_size)
            val_dataset = x_input_val['x_0'].batch(batch_size)

        
        print("Dataset: ", dataset)
        # save model config
        checkpoint_path = model_name + 'counting-fast' + '_checkpoint'

        early_stopings = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0.1, patience=10, verbose=1, mode="min", 
        )
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_loss", save_best_only=True, mode="min", verbose=0, save_weights_only=True
        )
        # early_stopings, 
        callbacks = [early_stopings, checkpoint]
        # fit
        history = model.fit(
            dataset,
            epochs=epochs,
            validation_data = val_dataset,
            callbacks=callbacks,
        )

        # model.build((None, 28, 28, 1))
        # model.save(model_path, save_format="tf")

        return history, model
    


    # def test(self, model_name, learning_rate, latent_dim, batch_size=256, epochs=50):
