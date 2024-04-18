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

    def command_decay(self, feature, t):
        """
        :param feature: (b,hidden_dim*4)=(b,256)
        :param t: simulation time_step
        :return: decayed feature
        """
        command_time = t % self.config.command_update_cycle
        discount = self.config.command_gamma ** command_time

        return feature * discount

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


class CommanderEncoder(tf.keras.models.Model):
    """
    Model: "commander_encoder"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     commander_cnn_model_1 (Comm  multiple                 258944
     anderCNNModel)

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

    @tf.function
    def call(self, inputs, time_step):
        """
        :param inputs: (b,commander_grid,commander_grid,
                        commander_observation_channels*commander_n_frames)
        :param time_step:
        :return: discounted feature
        """
        command_feature = self.commander_cnn(inputs)  # (b,4*hidden_dim)=(b,256)
        discounted_feature = self.commander_cnn.command_decay(command_feature, time_step)

        return discounted_feature


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

    for t in range(11):
        discounted_feature = encoder(states, t)
        print(t, discounted_feature[0, 0])

    encoder.summary()


if __name__ == "__main__":
    main()
