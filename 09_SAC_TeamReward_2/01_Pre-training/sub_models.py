import os.path

import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

from config import Config
from utils_transformer import make_mask, make_padded_obs


class CNNModel(tf.keras.models.Model):
    """
    # Add AutoEncoder
    :param obs_shape: (15,15,6), grid=15, ch=6, n_frames=1
    :param max_num_agents=n=15
    :return: (None,n,hidden_dim)=(None,15,256)

    Model: "cnn_model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     input_1 (InputLayer)        [(None, 15, 15, 15, 6)]   0

     time_distributed (TimeDistr  (None, 15, 15, 15, 64)   448
     ibuted)

     time_distributed_1 (TimeDis  (None, 15, 7, 7, 64)     36928
     tributed)

     time_distributed_2 (TimeDis  (None, 15, 5, 5, 64)     36928
     tributed)

     time_distributed_3 (TimeDis  (None, 15, 3, 3, 64)     36928
     tributed)

     time_distributed_4 (TimeDis  (None, 15, 576)          0
     tributed)

     time_distributed_5 (TimeDis  (None, 15, 256)          147712
     tributed)

     tf.math.multiply (TFOpLambd  (None, 15, 256)          0
     a)

    =================================================================
    Total params: 258,944
    Trainable params: 258,944
    Non-trainable params: 0
    _________________________________________________________________
    """

    def __init__(self, config, **kwargs):
        super(CNNModel, self).__init__(**kwargs)

        self.config = config

        self.conv0 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=1,
                    strides=1,
                    activation='relu',
                    kernel_initializer='Orthogonal'
                )
            )

        self.conv1 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=3,
                    strides=2,
                    activation='relu',
                    kernel_initializer='Orthogonal'
                )
            )

        self.conv2 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=3,
                    strides=1,
                    activation='relu',
                    kernel_initializer='Orthogonal'
                )
            )

        self.conv3 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=3,
                    strides=1,
                    activation='relu',
                    kernel_initializer='Orthogonal'
                )
            )

        self.flatten1 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Flatten()
            )

        self.dense1 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units=self.config.hidden_dim,
                    activation=None
                )
            )

    @tf.function
    def call(self, inputs, mask):
        # inputs: (b,n,g,g,ch*n_frames)=(1,15,15,15,6)
        # mask: (b,n)=(1,15), bool

        h = self.conv0(inputs)  # (1,15,20,20,64)
        h = self.conv1(h)  # (1,15,7,7,64)
        h = self.conv2(h)  # (1,15,5,5,64)
        h = self.conv3(h)  # (1,15,3,3,64)

        h1 = self.flatten1(h)  # (1,15,576)

        features = self.dense1(h1)  # (1,15,256)

        broadcast_float_mask = \
            tf.expand_dims(
                tf.cast(mask, 'float32'),
                axis=-1
            )  # Add feature dim for broadcast, (1,15,1)

        features = features * broadcast_float_mask  # (1,15,256)

        return features

    def build_graph(self, mask):
        """ For summary & plot_model """
        x = tf.keras.layers.Input(
            shape=(self.config.max_num_red_agents,
                   self.config.grid_size,
                   self.config.grid_size,
                   self.config.observation_channels * self.config.n_frames)
        )

        model = \
            tf.keras.models.Model(
                inputs=[x],
                outputs=self.call(x, mask),
                name='cnn_model'
            )

        return model


class MultiHeadAttentionModel(tf.keras.models.Model):
    """
    Two layers of MultiHeadAttention (Self Attention with provided mask)

    :param mask: (None,n,n), bool
    :param max_num_agents=15=n
    :param hidden_dim = 256

    :return: features: (None,n,hidden_dim)=(None,15,256)
             score: (None,num_heads,n,n)=(None,2,2,15)

    Model: "mha_1"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to
    ==================================================================================================
     input_3 (InputLayer)           [(None, 15, 256)]    0           []

     multi_head_attention (MultiHea  ((None, 15, 256),   263168      ['input_3[0][0]',
     dAttention)                     (None, 2, 15, 15))               'input_3[0][0]',
                                                                      'input_3[0][0]']

     add (Add)                      (None, 15, 256)      0           ['input_3[0][0]',
                                                                      'multi_head_attention[0][0]']

     dense_1 (Dense)                (None, 15, 512)      131584      ['add[0][0]']

     dense_2 (Dense)                (None, 15, 256)      131328      ['dense_1[0][0]']

     dropout (Dropout)              (None, 15, 256)      0           ['dense_2[0][0]']

     add_1 (Add)                    (None, 15, 256)      0           ['add[0][0]',
                                                                      'dropout[0][0]']

     tf.math.multiply_2 (TFOpLambda  (None, 15, 256)     0           ['add_1[0][0]']
     )

    ==================================================================================================
    Total params: 526,080
    Trainable params: 526,080
    Non-trainable params: 0
    __________________________________________________________________________________________________
    """

    def __init__(self, config, **kwargs):
        super(MultiHeadAttentionModel, self).__init__(**kwargs)

        self.config = config

        self.mha1 = \
            tf.keras.layers.MultiHeadAttention(
                num_heads=self.config.num_heads,
                key_dim=self.config.key_dim,
            )

        self.add1 = \
            tf.keras.layers.Add()

        """
        self.layernorm1 = \
            tf.keras.layers.LayerNormalization(
                axis=-1, center=True, scale=True
            )
        """

        self.dense1 = \
            tf.keras.layers.Dense(
                units=config.hidden_dim * 2,
                activation='relu',
            )

        self.dense2 = \
            tf.keras.layers.Dense(
                units=config.hidden_dim,
                activation=None,
            )

        self.dropoout1 = tf.keras.layers.Dropout(rate=self.config.dropout_rate)

        self.add2 = tf.keras.layers.Add()

        """
        self.layernorm2 = \
            tf.keras.layers.LayerNormalization(
                axis=-1, center=True, scale=True
            )
        """

    @tf.function
    def call(self, inputs, mask=None, training=True):
        # inputs: (None,n,hidden_dim)=(None,15,256)
        # mask: (None,n)=(None,15), bool

        float_mask = \
            tf.expand_dims(
                tf.cast(mask, 'float32'),
                axis=-1
            )  # (None,n,1)

        attention_mask = tf.matmul(
            float_mask, float_mask, transpose_b=True
        )  # (None,n,n)

        attention_mask = tf.cast(attention_mask, 'bool')

        x, score = \
            self.mha1(
                query=inputs,
                key=inputs,
                value=inputs,
                attention_mask=attention_mask,
                return_attention_scores=True,
            )  # (None,n,hidden_dim),(None,num_heads,n,n)=(None,15,256),(None,2,15,15)

        x1 = self.add1([inputs, x])  # (None,n,hidden_dim)=(None,15,256)

        # x1 = self.layernorm1(x1)

        x2 = self.dense1(x1)  # (None,n,hidden_dim)=(None,15,512)

        x2 = self.dense2(x2)  # (None,n,hidden_dim)=(None,15,256)

        x2 = self.dropoout1(x2, training=training)

        features = self.add2([x1, x2])  # (None,n,hidden_dim)=(None,15,256)

        # features = self.layernorm2(features)

        broadcast_float_mask = \
            tf.expand_dims(
                tf.cast(mask, 'float32'),
                axis=-1
            )  # Add feature dim for broadcast, (None,n,1)=(1,15,1)

        features = features * broadcast_float_mask

        return features, score

    def build_graph(self, mask, idx):
        x = tf.keras.layers.Input(
            shape=(self.config.max_num_red_agents,
                   self.config.hidden_dim)
        )

        model = tf.keras.models.Model(
            inputs=[x],
            outputs=self.call(x, mask, training=True),
            name='mha_' + str(idx),
        )

        return model


class PolicyModel(tf.keras.models.Model):
    """
    :param action_dim=5
    :param hidden_dim=256
    :param max_num_agents=15=n
    :return: Policy probs, (None,n,action_dim)=(None,15,5)

    Model: "policy_logits"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     input_1 (InputLayer)        [(None, 15, 256)]         0

     time_distributed_6 (TimeDis  (None, 15, 768)          197376
     tributed)

     time_distributed_8 (TimeDis  (None, 15, 256)          196864
     tributed)

     time_distributed_9 (TimeDis  (None, 15, 5)            1285
     tributed)

     tf.math.multiply (TFOpLambd  (None, 15, 5)            0
     a)

    =================================================================
    Total params: 395,525
    Trainable params: 395,525
    Non-trainable params: 0
    _________________________________________________________________
    """

    def __init__(self, config, **kwargs):
        super(PolicyModel, self).__init__(**kwargs)

        self.config = config

        self.dense1 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units=self.config.hidden_dim * 3,
                    activation='relu',
                )
            )

        self.dropoout1 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dropout(rate=self.config.dropout_rate)
        )

        self.dense2 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units=self.config.hidden_dim,
                    activation='relu',
                )
            )

        self.dense3 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units=self.config.action_dim,
                    activation=None,
                )
            )

    @tf.function
    def call(self, inputs, mask, training=True):
        # inputs: (None,n,hidden_dim)=(None,15,256)
        # mask: (None,n)=(None,15), bool
        # prob_logit=0 for dead / dummy agents

        x1 = self.dense1(inputs)  # (None,n,hidden_dim*3)

        # x1 = self.dropoout1(x1, training=training)

        x1 = self.dense2(x1)  # (None,n,hidden_dim)

        logits = self.dense3(x1)  # (None,n,action_dim)

        # mask out dead or dummy agents by 0
        broadcast_float_mask = \
            tf.expand_dims(
                tf.cast(mask, 'float32'),
                axis=-1
            )  # Add feature dim for broadcast, (None,n,1)=(None,15,1)

        logits = logits * broadcast_float_mask  # (None,n,action_dim)

        return logits

    def build_graph(self, mask):
        x = tf.keras.layers.Input(
            shape=(self.config.max_num_red_agents,
                   self.config.hidden_dim)
        )  # (None,n,256)

        model = tf.keras.models.Model(
            inputs=[x],
            outputs=self.call(x, mask),
            name='policy_logits'
        )

        return model


class SoftQModel(tf.keras.models.Model):
    """
    :param action_dim=5
    :param hidden_dim=256
    :param max_num_agents=15=n
    :return: soft_Qs, (None,n,action_dim)=(None,15,5)

    Model: "value_model"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to
    ==================================================================================================
     input_1 (InputLayer)           [(None, 15, 256)]    0           []

     time_distributed_11 (TimeDistr  (None, 15, 768)     197376      ['input_1[0][0]']
     ibuted)

     time_distributed_14 (TimeDistr  (None, 15, 768)     197376      ['input_1[0][0]']
     ibuted)

     time_distributed_12 (TimeDistr  (None, 15, 256)     196864      ['time_distributed_11[0][0]']
     ibuted)

     time_distributed_15 (TimeDistr  (None, 15, 256)     196864      ['time_distributed_14[0][0]']
     ibuted)

     time_distributed_13 (TimeDistr  (None, 15, 5)       1285        ['time_distributed_12[0][0]']
     ibuted)

     time_distributed_16 (TimeDistr  (None, 15, 5)       1285        ['time_distributed_15[0][0]']
     ibuted)

     tf.math.multiply (TFOpLambda)  (None, 15, 5)        0           ['time_distributed_13[0][0]']

     tf.math.multiply_1 (TFOpLambda  (None, 15, 5)       0           ['time_distributed_16[0][0]']
     )

    ==================================================================================================
    Total params: 791,050
    Trainable params: 791,050
    Non-trainable params: 0
    __________________________________________________________________________________________________

    """

    def __init__(self, config, **kwargs):
        super(SoftQModel, self).__init__(**kwargs)

        self.config = config
        self.action_dim = config.action_dim

        # For Q1
        self.dense1 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units=self.config.hidden_dim * 3,
                    activation='relu',
                )
            )

        # self.dropoout1 = tf.keras.layers.TimeDistributed(
        #     tf.keras.layers.Dropout(rate=self.config.dropout_rate)
        # )

        self.dense2 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units=self.config.hidden_dim,
                    activation='relu',
                )
            )

        self.dense3 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units=self.action_dim,
                    activation=None,
                )
            )

        # For Q2
        self.dense10 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units=self.config.hidden_dim * 3,
                    activation='relu',
                )
            )

        # self.dropoout10 = tf.keras.layers.TimeDistributed(
        #     tf.keras.layers.Dropout(rate=self.config.dropout_rate)
        # )

        self.dense20 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units=self.config.hidden_dim,
                    activation='relu',
                )
            )

        self.dense30 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units=self.action_dim,
                    activation=None,
                )
            )

    @tf.function
    def call(self, inputs, mask, training=True):
        """
        Q1, Q2 = 0 for dead / dummy agents
        """

        """ mask out dead or dummy agents by 0 """
        broadcast_float_mask = \
            tf.expand_dims(
                tf.cast(mask, 'float32'),
                axis=-1
            )  # Add feature dim for broadcast, (None,n,1)

        """ Q1 """
        x1 = self.dense1(inputs)  # (None,n,hidden_dim)
        # x1 = self.dropoout1(x1, training=training)
        x1 = self.dense2(x1)
        qs1 = self.dense3(x1)  # (None,n,action_dim)

        qs1 = qs1 * broadcast_float_mask  # (None,n,action_dim)

        """ Q2 """
        x10 = self.dense10(inputs)  # (None,n,hidden_dim)
        # x10 = self.dropoout10(x10, training=training)
        x10 = self.dense20(x10)
        qs10 = self.dense30(x10)  # (None,n,action_dim)

        qs10 = qs10 * broadcast_float_mask  # (None,n,action_dim)

        return qs1, qs10

    def build_graph(self, mask):
        x = tf.keras.layers.Input(
            shape=(self.config.max_num_red_agents,
                   self.config.hidden_dim)
        )  # (None,n,256)

        model = tf.keras.models.Model(
            inputs=[x],
            outputs=self.call(x, mask),
            name='value_model'
        )

        return model


def main():
    dir_name = './models_architecture'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    config = Config()

    grid_size = config.grid_size
    ch = config.observation_channels
    n_frames = config.n_frames

    obs_shape = (grid_size, grid_size, ch * n_frames)

    max_num_agents = config.max_num_red_agents

    # Define alive_agents_ids & raw_obs
    alive_agents_ids = [0, 2]
    agent_obs = {}

    for i in alive_agents_ids:
        agent_id = 'red_' + str(i)
        agent_obs[agent_id] = np.ones(obs_shape)

    # Get padded_obs and mask
    padded_obs = make_padded_obs(max_num_agents, obs_shape, agent_obs)  # (1,n,g,g,ch*n_frames)

    mask = make_mask(alive_agents_ids, max_num_agents)  # (1,n)

    """ cnn_model """
    cnn = CNNModel(config=config)

    features_cnn = cnn(padded_obs, mask)  # Build, (1,n,hidden_dim)

    """ remove tf.function for summary """
    """
    cnn.build_graph(mask).summary()

    tf.keras.utils.plot_model(
        cnn.build_graph(mask),
        to_file=dir_name + '/cnn_model',
        show_shapes=True,
        show_layer_activations=True,
        show_dtype=True,
        dpi=96 * 3
    )
    """

    """ mha model """
    mha = MultiHeadAttentionModel(config=config)

    features_mha, score = mha(features_cnn, mask)  # Build, (None,n,hidden_dim),(1,num_heads,n,n)

    """ remove tf.function for summary """
    """
    idx = 1
    mha.build_graph(mask, idx).summary()

    tf.keras.utils.plot_model(
        mha.build_graph(mask, idx),
        to_file=dir_name + '/mha_model_' + str(idx),
        show_shapes=True,
        show_layer_activations=True,
        show_dtype=True,
        dpi=96 * 3
    )
    """

    """ policy_model """

    policy_model = PolicyModel(config=config)

    policy_logits = policy_model(features_mha, mask)
    print(f'policy_logits.shape: {policy_logits.shape}, {policy_logits}')

    """ remove tf.function for summary """
    """
    policy_model.build_graph(mask).summary()

    tf.keras.utils.plot_model(
        policy_model.build_graph(mask),
        to_file=dir_name + '/policy_model',
        show_shapes=True,
        show_layer_activations=True,
        show_dtype=True,
        dpi=96 * 3
    )
    """

    """ value_model """

    q_model = SoftQModel(config=config)

    qs1, qs2 = q_model(features_mha, mask)
    print(f'Q1s.shape: {qs1.shape}, Q2s.shape: {qs2.shape}')

    """ remove tf.function for summary """
    """
    q_model.build_graph(mask).summary()

    tf.keras.utils.plot_model(
        q_model.build_graph(mask),
        to_file=dir_name + '/q_model',
        show_shapes=True,
        show_layer_activations=True,
        show_dtype=True,
        dpi=96 * 3
    )
    """


if __name__ == '__main__':
    main()
