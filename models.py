import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import  layers, Input, Model, metrics


class AutoEncoder(tf.keras.Model):
    def __init__(self, latent_dim, image_target_height, image_target_width):
        super().__init__()

        self.latent_dim = latent_dim
        self.image_target_height = image_target_height
        self.image_target_width = image_target_width
        # Encoder
        self.encoder_reshape = layers.Reshape(
            (image_target_height * image_target_width,)
        )  # Shape as 64,64,1
        self.encoder_fc1 = layers.Dense(32, activation="relu")
        self.encoder_fc2 = layers.Dense(latent_dim, activation="relu")

        # Decoder
        self.decoder_fc1 = layers.Dense(32, activation="relu")
        self.decoder_fc2 = layers.Dense(
            image_target_height * image_target_width, activation="sigmoid"
        )
        self.decoder_reshape = layers.Reshape(
            (image_target_height, image_target_width, 1)
        )

        self._build_graph()

    def _build_graph(self):
        input_shape = (self.image_target_height, self.image_target_width, 1)
        self.build((None,) + input_shape)
        inputs = tf.keras.Input(shape=input_shape)
        _ = self.call(inputs)

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


def encode(latent_dim, image_target_height, image_target_width):
    encoder_inputs = Input(shape=(image_target_height, image_target_width, 1))
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(
        encoder_inputs
    )
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var", activation="relu")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()
    return encoder


def decode(latent_dim):
    latent_inputs = Input(shape=(latent_dim,))
    x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(
        1, 3, activation="sigmoid", padding="same"
    )(x)
    decoder = Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()

    return decoder


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
