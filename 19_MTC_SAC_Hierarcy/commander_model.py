import os
import numpy as np
import tensorflow as tf

from config_hierarcy import Config


class CommanderCNNModel(tf.keras.models.Model):
    """
    Model: "commander_cnn_model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     conv2d (Conv2D)             multiple                  448

     conv2d_1 (Conv2D)           multiple                  36928

     conv2d_2 (Conv2D)           multiple                  36928

     conv2d_3 (Conv2D)           multiple                  36928

     flatten (Flatten)           multiple                  0

     dense (Dense)               multiple                  147712

    =================================================================
    Total params: 258,944
    Trainable params: 258,944
    Non-trainable params: 0
    _________________________________________________________________
    """

    def __init__(self, config, **kwargs):
        super(CommanderCNNModel, self).__init__(**kwargs)

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

        self.dense1 = tf.keras.layers.Dense(
            units=config.hidden_dim * 4,  # hidden_dim=64
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

        feature = self.dense1(feature)  # (b,256)

        return feature

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


class ElapsedTimeEncoder:
    def __init__(self, config, **kwargs):
        # depth = hidden_dim * 4 = 256
        # h_dim = depth/2 = 128

        super(ElapsedTimeEncoder, self).__init__(**kwargs)

        self.config = config

        depth = self.config.hidden_dim * 4  # 256
        h_dim = depth / 2

        depths = np.arange(h_dim) / h_dim  # (h_dim,)
        depths = np.expand_dims(depths, axis=0)  # (1,h_dim)

        self.angle_rates = 1 / (1000 ** depths)  # (1,h_dim)

    def call(self, elapsed_time):
        # elapsed_time: (b,1)
        # return: encoded_elapsed_time: (b,hidden_dim*4)=(b,256)

        angle_rads = elapsed_time * self.angle_rates  # (b, h_dim)

        encoded_elapsed_time = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)],
                                              axis=-1)  # (b,2*h_dim)=(b,depth)=(b,4*hidden_dim)

        encoded_elapsed_time = tf.cast(encoded_elapsed_time, dtype=tf.float32)

        return encoded_elapsed_time  # (b,256)


class CommanderEncoder(tf.keras.models.Model):
    """
    Model: "commander_encoder"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     commander_cnn_model_1 (Comm  multiple                 258944
     anderCNNModel)

     add (Add)                   multiple                  0

    =================================================================
    Total params: 258,944
    Trainable params: 258,944
    Non-trainable params: 0
    _________________________________________________________________
    """

    def __init__(self, config, **kwargs):
        super(CommanderEncoder, self).__init__(**kwargs)

        self.config = config
        self.commander_cnn = CommanderCNNModel(self.config)
        self.elapsed_time_encoder = ElapsedTimeEncoder(self.config)
        self.add1 = tf.keras.layers.Add()

    def call(self, inputs, elapsed_time):
        """
        :param inputs: (b,commander_grid,commander_grid,
                        commander_observation_channels*commander_n_frames)
        :param elapsed_time:  (b,1)
        :return: commander feature
        """
        cnn_feature = self.commander_cnn(inputs)  # (b,4*hidden_dim)=(b,256)
        encoded_elapsed_time = self.elapsed_time_encoder.call(elapsed_time)
        # (b,4*hidden_dim)=(b,256)

        scale = tf.math.sqrt(tf.cast(self.config.hidden_dim * 4, tf.float32))

        commander_feature = self.add1([cnn_feature * scale, encoded_elapsed_time])  # (b,256)

        return commander_feature


def main():
    dir_name = './models_architecture'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    config = Config()

    """ check conv_network """
    states = np.random.rand(config.batch_size,
                            config.commander_grid_size, config.commander_grid_size,
                            config.commander_observation_channels * config.commander_n_frames)
    cnn = CommanderCNNModel(config)

    feature = cnn(states)  # (b, 4*hidden_dim)=(b,256)
    print(feature.shape)

    cnn.summary()

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

    encoder = CommanderEncoder(config)

    elapsed_time = np.random.randint(low=0, high=10, size=(config.batch_size, 1))  # (16,1)

    commander_feature = encoder(states, elapsed_time)
    print(elapsed_time[0, 0], commander_feature[0, 0])

    encoder.summary()


if __name__ == "__main__":
    main()
