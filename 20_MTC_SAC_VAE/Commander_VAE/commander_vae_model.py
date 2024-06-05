import os
import numpy as np
import tensorflow as tf

from config_for_vae_training import Config


class VAEEncoder(tf.keras.models.Model):
    """
    Model: "vae_encoder"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     conv2d (Conv2D)             multiple                  448

     conv2d_1 (Conv2D)           multiple                  36928

     conv2d_2 (Conv2D)           multiple                  36928

     conv2d_3 (Conv2D)           multiple                  36928

     flatten (Flatten)           multiple                  0

     dense (Dense)               multiple                  73856

     dense_1 (Dense)             multiple                  73856

    =================================================================
    Total params: 258,944
    Trainable params: 258,944
    Non-trainable params: 0
    _________________________________________________________________
    """

    def __init__(self, config, **kwargs):
        super(VAEEncoder, self).__init__(**kwargs)

        self.config = config

        self.conv0 = \
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=1,
                strides=1,
                activation='relu',
                kernel_initializer='Orthogonal'
            )

        self.conv1 = \
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=3,
                strides=2,
                activation='relu',
                kernel_initializer='Orthogonal'
            )

        self.conv2 = \
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=3,
                strides=2,
                activation='relu',
                kernel_initializer='Orthogonal'
            )

        self.conv3 = \
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=3,
                strides=1,
                activation='relu',
                kernel_initializer='Orthogonal'
            )

        self.flatten1 = tf.keras.layers.Flatten()

        self.dense_mean1 = tf.keras.layers.Dense(
            units=config.hidden_dim * self.config.latent_mult,  # hidden_dim=64
            activation=None
        )

        self.dense_log_var1 = tf.keras.layers.Dense(
            units=config.hidden_dim * self.config.latent_mult,  # hidden_dim=64
            activation=None
        )

    @tf.function
    def call(self, inputs):
        # inputs: (b,commander_g,commander_g,commander_ch*commander_n_frames)=(b,25,25,6)

        h = self.conv0(inputs)  # (b,25,25,64)
        h = self.conv1(h)  # (b,12,12,64)
        h = self.conv2(h)  # (b,5,5,64)
        h = self.conv3(h)  # (b,3,3,64)

        feature = self.flatten1(h)  # (b,576)

        feature_mean = self.dense_mean1(feature)  # (b,128)
        feature_log_var = self.dense_log_var1(feature)  # (b,128)

        return feature_mean, feature_log_var

    def build_graph(self):
        """ For summary & plot_model """
        x = tf.keras.layers.Input(
            shape=(self.config.commander_grid_size,
                   self.config.commander_grid_size,
                   self.config.commander_observation_channels * self.config.commander_n_frames)
        )

        model = \
            tf.keras.models.Model(
                inputs=[x],
                outputs=self.call(x),
                name='commander_cnn_model'
            )

        return model


class SampleFeature(tf.keras.models.Model):
    """
    Model: "sample_feature"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     lambda (Lambda)             multiple                  0

    =================================================================
    Total params: 0
    Trainable params: 0
    Non-trainable params: 0
    _________________________________________________________________
    """

    def __init__(self, config, **kwargs):
        super(SampleFeature, self).__init__(**kwargs)

        self.config = config

        def sampling(args):
            feature_mean, feature_log_var = args
            (batch, dim) = feature_mean.shape
            epsilon = tf.random.normal(shape=(batch, dim))
            feature = feature_mean + tf.math.exp(0.5 * feature_log_var) * epsilon
            return feature

        self.sampling_lambda = tf.keras.layers.Lambda(
            sampling,
            output_shape=self.config.hidden_dim * self.config.latent_mult,
        )

    @tf.function
    def call(self, inputs):
        feature_mean = inputs[0]  # (b,128)
        feature_log_var = inputs[1]  # (b,128)

        z = self.sampling_lambda([feature_mean, feature_log_var])  # (b,128)

        return z


class VAEDecoder(tf.keras.models.Model):
    """
    Model: "vae_decoder"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     dense_2 (Dense)             multiple                  74304

     reshape (Reshape)           multiple                  0

     conv2d_transpose (Conv2DTra  multiple                 36928
     nspose)

     conv2d_transpose_1 (Conv2DT  multiple                 65600
     ranspose)

     conv2d_transpose_2 (Conv2DT  multiple                 36928
     ranspose)

     conv2d_transpose_3 (Conv2DT  multiple                 390
     ranspose)

    =================================================================
    Total params: 214,150
    Trainable params: 214,150
    Non-trainable params: 0
    _________________________________________________________________
    """

    def __init__(self, config, **kwargs):
        super(VAEDecoder, self).__init__(**kwargs)

        self.config = config

        self.dense1 = tf.keras.layers.Dense(units=3 * 3 * 64, activation=None)

        self.reshape1 = tf.keras.layers.Reshape(target_shape=(3, 3, -1))

        self.deconv0 = tf.keras.layers.Conv2DTranspose(
            filters=64,
            kernel_size=3,
            strides=1,
            activation='relu',
            kernel_initializer='Orthogonal'
        )

        self.deconv1 = tf.keras.layers.Conv2DTranspose(
            filters=64,
            kernel_size=4,
            strides=2,
            activation='relu',
            kernel_initializer='Orthogonal'
        )

        self.deconv2 = tf.keras.layers.Conv2DTranspose(
            filters=64,
            kernel_size=3,
            strides=2,
            activation='relu',
            kernel_initializer='Orthogonal'
        )

        self.deconv3 = tf.keras.layers.Conv2DTranspose(
            filters=6,
            kernel_size=1,
            strides=1,
            activation='sigmoid',
            kernel_initializer='Orthogonal'
        )

    @tf.function
    def call(self, inputs):
        # inputs: (b,hidden_dim*1)=(b,128)

        h = self.dense1(inputs)  # (b,576)
        h = self.reshape1(h)  # (b,3,3,64)
        h = self.deconv0(h)  # (b,5,5,64)
        h = self.deconv1(h)  # (b,12,12,64)
        h = self.deconv2(h)  # (b,25,25,64)
        imgs = self.deconv3(h)  # (b,25,25,6)

        return imgs


class CommanderVAE(tf.keras.models.Model):
    """
    Model: "commander_vae"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     vae_encoder_1 (VAEEncoder)  multiple                  258944

     sample_feature_1 (SampleFea  multiple                 0
     ture)

     vae_decoder_1 (VAEDecoder)  multiple                  214150

    =================================================================
    Total params: 473,094
    Trainable params: 473,094
    Non-trainable params: 0
    _________________________________________________________________
    """

    def __init__(self, config, **kwargs):
        super(CommanderVAE, self).__init__(**kwargs)

        self.config = config

        self.encoder = VAEEncoder(self.config)
        self.sampler = SampleFeature(self.config)
        self.decoder = VAEDecoder(self.config)

    @tf.function
    def call(self, inputs):
        # inputs: (b,25,25,6)
        z_mean, z_log_var = self.encoder(inputs)  # (b,128),(b,128)
        z = self.sampler([z_mean, z_log_var])  # (b,128)
        reconst_maps = self.decoder(z)  # (b,25,25,6)

        return reconst_maps, z_mean, z_log_var


class ElapsedTimeEncoder:
    def __init__(self, config, **kwargs):
        # depth = hidden_dim * 2 = 128
        # h_dim = depth/2 = 128

        super(ElapsedTimeEncoder, self).__init__(**kwargs)

        self.config = config

        depth = self.config.hidden_dim * 2  # 128
        h_dim = depth / 2

        depths = np.arange(h_dim) / h_dim  # (h_dim,)
        depths = np.expand_dims(depths, axis=0)  # (1,h_dim)

        self.angle_rates = 1 / (1.05 ** depths)  # (1,h_dim)

    def call(self, elapsed_time):
        # elapsed_time: (b,1)
        # return: encoded_elapsed_time: (b,hidden_dim*2)=(b,128)

        angle_rads = elapsed_time * self.angle_rates  # (b, h_dim)=(b,64)

        encoded_elapsed_time = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)],
                                              axis=-1)  # (b,2*h_dim)=(b,depth)=(b,2*hidden_dim)

        encoded_elapsed_time = tf.cast(encoded_elapsed_time, dtype=tf.float32)

        return encoded_elapsed_time  # (b,128)


class SampleFeatureTester(tf.keras.models.Model):
    def __init__(self, config, **kwargs):
        super(SampleFeatureTester, self).__init__(**kwargs)

        self.config = config

    @tf.function
    def call(self, inputs):
        feature_mean = inputs[0]  # (b,128)
        return feature_mean


class CommanderEncoder(tf.keras.models.Model):
    """
    Model: "commander_encoder"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     commander_vae_1 (CommanderV  multiple                 473094
     AE)

     sample_feature_3 (SampleFea  multiple                 0
     ture)

     concatenate (Concatenate)   multiple                  0

    =================================================================
    Total params: 473,094
    Trainable params: 473,094
    Non-trainable params: 0
    _________________________________________________________________
    """

    def __init__(self, config, **kwargs):
        super(CommanderEncoder, self).__init__(**kwargs)

        self.config = config

        self.commander_vae = CommanderVAE(self.config)

        self.sampler = SampleFeatureTester(self.config)

        self.elapsed_time_encoder = ElapsedTimeEncoder(self.config)

        self.concate1 = tf.keras.layers.Concatenate(axis=-1)

    def load_weights(self, model_name):
        self.commander_vae.load_weights(model_name)

    def call(self, inputs, elapsed_time):
        """
        :param inputs: (b,commander_grid,commander_grid,
                        commander_observation_channels*commander_n_frames)
        :param elapsed_time:  (b,1)
        :return: commander feature
        """
        _, z_mean, z_log_var = self.commander_vae(inputs)  # (b,128), (b,128)
        z = self.sampler([z_mean, z_log_var])  # (b,128)

        encoded_elapsed_time = self.elapsed_time_encoder.call(elapsed_time)
        # (b,2*hidden_dim)=(b,128)

        scale = tf.math.sqrt(tf.cast(self.config.hidden_dim * self.config.latent_mult, tf.float32))

        commander_feature = self.concate1([z * scale, encoded_elapsed_time])  # (b,128)

        return commander_feature  # (b,256)


def main():
    dir_name = './models_architecture'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    config = Config()

    """ vae encoder """
    states = np.random.rand(config.batch_size,
                            config.commander_grid_size, config.commander_grid_size,
                            config.commander_observation_channels * config.commander_n_frames)
    encoder = VAEEncoder(config)

    z_mean, z_log_var = encoder(states)  # (b, 4*hidden_dim)=(b,256)

    encoder.summary()

    """
    cnn.build_graph().summary()

    tf.keras.utils.plot_model(
        cnn.build_graph(),
        to_file=dir_name + '/commander_cnn_model',
        show_shapes=True,
        show_layer_activations=True,
        show_dtype=True,
        dpi=96 * 3
    )
    """

    """ vae sampler """
    sample_feature = SampleFeature(config)
    z = sample_feature([z_mean, z_log_var])
    sample_feature.summary()

    """ vae decoder """
    decoder = VAEDecoder(config)
    reconst_imgs = decoder(z)  # (b,25,25,6)
    decoder.summary()

    """ vae """
    commander_vae = CommanderVAE(config)
    reconst_maps, z_mean, z_log_var = commander_vae(states)  # (b,25,25,6)
    commander_vae.summary()

    """ elapsed-time encoder """
    elapsed_time = np.random.randint(low=0, high=10, size=(config.batch_size, 1))  # (16,1)
    elapsed_time_encoder = ElapsedTimeEncoder(config)
    elapsed_time_encoder.call(elapsed_time)

    """ commander encoder """
    encoder = CommanderEncoder(config)

    model_name = './vaes/vae_1400/'
    encoder.load_weights(model_name)

    commander_feature = encoder(states, elapsed_time)
    print(elapsed_time.shape, commander_feature.shape)

    encoder.summary()

    """ sampler for test """
    sampler_for_test = SampleFeatureTester(config)
    commander_feature = sampler_for_test([z_mean, z_log_var])
    print(commander_feature.shape)

    sampler_for_test.summary()


if __name__ == "__main__":
    main()
