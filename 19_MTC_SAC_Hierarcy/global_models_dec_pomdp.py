import os
import numpy as np
import tensorflow as tf

from config_hierarcy import Config


class GlobalCNNModel(tf.keras.models.Model):
    """
    Model: "global_cnn_model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     input_1 (InputLayer)        [(None, 15, 15, 6)]       0

     conv2d (Conv2D)             (None, 15, 15, 64)        448

     conv2d_1 (Conv2D)           (None, 7, 7, 64)          36928

     conv2d_2 (Conv2D)           (None, 5, 5, 64)          36928

     conv2d_3 (Conv2D)           (None, 3, 3, 64)          36928

     flatten (Flatten)           (None, 576)               0

     dense (Dense)               (None, 256)               147712

    =================================================================
    Total params: 258,944
    Trainable params: 258,944
    Non-trainable params: 0
    _________________________________________________________________
    """

    def __init__(self, config, **kwargs):
        super(GlobalCNNModel, self).__init__(**kwargs)

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
                strides=1,
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
        # inputs: (b,g,g,global_ch*global_n_frames)=(b,15,15,6)

        h = self.conv0(inputs)  # (b,15,15,64)
        h = self.conv1(h)  # (b,7,7,64)
        h = self.conv2(h)  # (b,5,5,64)
        h = self.conv3(h)  # (b,3,3,64)

        feature = self.flatten1(h)  # (b,576)

        feature = self.dense1(feature)  # (b,256)

        return feature

    def build_graph(self):
        """ For summary & plot_model """
        x = tf.keras.layers.Input(
            shape=(self.config.grid_size,
                   self.config.grid_size,
                   self.config.global_observation_channels * self.config.global_n_frames)
        )

        model = \
            tf.keras.models.Model(
                inputs=[x],
                outputs=self.call(x),
                name='global_cnn_model'
            )

        return model


def main():
    dir_name = './models_architecture'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    config = Config()

    """ check conv_network """
    states = np.random.rand(config.batch_size, config.grid_size, config.grid_size,
                            config.global_observation_channels * config.global_n_frames)
    cnn = GlobalCNNModel(config)

    feature = cnn(states)  # (b, hidden_dim)
    print(feature.shape)

    """
    cnn.build_graph().summary()

    tf.keras.utils.plot_model(
        cnn.build_graph(),
        to_file=dir_name + '/global_cnn_model',
        show_shapes=True,
        show_layer_activations=True,
        show_dtype=True,
        dpi=96 * 3
    )
    """


if __name__ == "__main__":
    main()
