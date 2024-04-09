import os.path

import keras.layers
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

from config_dec_pomdp import Config
from utils_transformer_mtc_dec_pomdp import make_mask, make_padded_obs, make_padded_pos
from global_models_dec_pomdp import GlobalCNNModel


class CNNModel(tf.keras.models.Model):
    """
    # Add AutoEncoder
    :param max_num_agents=n=15
    :inputs: [obs, pos], obs: (None,n,2*fov+1,2*fov+1,ch*n_frames), pos: (None,n,2*n_frames)
    :return: (None,n,hidden_dim)=(None,15,64)

    Model: "cnn_model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     time_distributed (TimeDistr  multiple                 1088
     ibuted)

     time_distributed_1 (TimeDis  multiple                 73856
     tributed)

     time_distributed_2 (TimeDis  multiple                 147584
     tributed)

     time_distributed_3 (TimeDis  multiple                 0
     tributed)

     time_distributed_4 (TimeDis  multiple                 288
     tributed)

     time_distributed_5 (TimeDis  multiple                 0
     tributed)

     time_distributed_6 (TimeDis  multiple                 10304
     tributed)

    =================================================================
    Total params: 233,120
    Trainable params: 233,120
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
                    filters=128,
                    kernel_size=3,
                    strides=1,
                    activation='relu',
                    kernel_initializer='Orthogonal'
                )
            )

        self.conv2 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    filters=128,
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

        self.dense_pos_enc = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(units=32,
                                      activation='relu',
                                      )
            )

        self.concatenate = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Concatenate(axis=-1)
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
        # inputs: [obs, pos],
        #   obs: (None,n,2*fov+1,2*fov+1,ch*n_frames)=(None,15,5,5,16),
        #   pos: (None,n,2*n_frames)=(None,15,8)
        # (alive_)mask: (b,n)=(1,15), bool

        h = self.conv0(inputs[0])  # (1,15,5,5,64)
        h = self.conv1(h)  # (1,15,3,3,128)
        h = self.conv2(h)  # (1,15,1,1,128)

        h1 = self.flatten1(h)  # (1,15,128)

        pos_enc = self.dense_pos_enc(inputs[1])  # (1,15,32)

        z = self.concatenate([h1, pos_enc])  # (1,15,160)

        features = self.dense1(z)  # (1,15,64)

        broadcast_float_mask = \
            tf.expand_dims(
                tf.cast(mask, 'float32'),
                axis=-1
            )  # Add feature dim for broadcast, (1,15,1)

        features = features * broadcast_float_mask  # (1,15,64)

        return features

    def build_graph(self, mask):
        """ For summary & plot_model """
        x0 = tf.keras.layers.Input(
            shape=(self.config.max_num_red_agents,
                   2 * self.config.fov + 1,
                   2 * self.config.fov + 1,
                   self.config.observation_channels * self.config.n_frames)
        )

        x1 = tf.keras.layers.Input(
            shape=(self.config.max_num_red_agents, 2 * self.config.n_frames)
        )

        x = [x0, x1]

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
    :param hidden_dim = 64

    :return: features: (None,n,hidden_dim)=(None,15,64)
             score: (None,num_heads,n,n)=(None,2,15,15)

    Model: "multi_head_attention_model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     multi_head_attention (Multi  multiple                 33216
     HeadAttention)

     add (Add)                   multiple                  0

     dense_3 (Dense)             multiple                  8320

     dense_4 (Dense)             multiple                  8256

     dropout (Dropout)           multiple                  0

     add_1 (Add)                 multiple                  0

    =================================================================
    Total params: 49,792
    Trainable params: 49,792
    Non-trainable params: 0
    _________________________________________________________________
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
    def call(self, inputs, mask=None, attention_mask=None, training=True):
        # inputs: (None,n,hidden_dim)=(None,15,64)
        # attention_mask: (None,n,n)=(None,15,15), bool

        x, score = \
            self.mha1(
                query=inputs,
                key=inputs,
                value=inputs,
                attention_mask=attention_mask,
                return_attention_scores=True,
            )  # (None,n,hidden_dim),(None,num_heads,n,n)=(None,15,64),(None,2,15,15)

        x1 = self.add1([inputs, x])  # (None,n,hidden_dim)=(None,15,64)

        # x1 = self.layernorm1(x1)

        x2 = self.dense1(x1)  # (None,n,2*hidden_dim)=(None,15,128)

        x2 = self.dense2(x2)  # (None,n,hidden_dim)=(None,15,64)

        x2 = self.dropoout1(x2, training=training)

        features = self.add2([x1, x2])  # (None,n,hidden_dim)=(None,15,64)

        # features = self.layernorm2(features)

        broadcast_float_mask = \
            tf.expand_dims(
                tf.cast(mask, 'float32'),
                axis=-1
            )  # Add feature dim for broadcast, (None,n,1)=(1,15,1)

        features = features * broadcast_float_mask

        return features, score

    def build_graph(self, mask, attention_mask, idx):
        x = tf.keras.layers.Input(
            shape=(self.config.max_num_red_agents,
                   self.config.hidden_dim)
        )

        model = tf.keras.models.Model(
            inputs=[x],
            outputs=self.call(x, mask, attention_mask, training=True),
            name='mha_' + str(idx),
        )

        return model


class PolicyModel(tf.keras.models.Model):
    """
    :param action_dim=5
    :param hidden_dim=64
    :param max_num_agents=15=n
    :return: Policy probs, (None,n,action_dim)=(None,15,5)

    Model: "policy_model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     time_distributed_7 (TimeDis  multiple                 12480
     tributed)

     time_distributed_8 (TimeDis  multiple                 0 (unused)
     tributed)

     time_distributed_9 (TimeDis  multiple                 12352
     tributed)

     time_distributed_10 (TimeDi  multiple                 325
     stributed)

    =================================================================
    Total params: 25,157
    Trainable params: 25,157
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
        # inputs: (None,n,hidden_dim)=(None,15,64)
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
        )  # (None,n,64)

        model = tf.keras.models.Model(
            inputs=[x],
            outputs=self.call(x, mask),
            name='policy_logits'
        )

        return model


class SoftQModelGlobalState(tf.keras.models.Model):
    """
    :param action_dim=5
    :param hidden_dim=64
    :param max_num_agents=15=n
    :inputs: [agents_feature, global_feature]=[(None,n,hidden_dim),(None,hidden_dim)]
    :return: soft_Qs, (None,n,action_dim)=(None,15,5)

    Model: "soft_q_model_global_state"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     time_distributed_11 (TimeDi  multiple                 61632
     stributed)

     time_distributed_12 (TimeDi  multiple                 12352
     stributed)

     time_distributed_13 (TimeDi  multiple                 325
     stributed)

     time_distributed_14 (TimeDi  multiple                 61632
     stributed)

     time_distributed_15 (TimeDi  multiple                 12352
     stributed)

     time_distributed_16 (TimeDi  multiple                 325
     stributed)

    =================================================================
    Total params: 148,618
    Trainable params: 148,618
    Non-trainable params: 0
    _________________________________________________________________
    """

    def __init__(self, config, **kwargs):
        super(SoftQModelGlobalState, self).__init__(**kwargs)

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

        inputs=[agents_feature, global_feature]=[(None,n,hidden_dim),(None,4*hidden_dim)]
        """

        """ mask out dead or dummy agents by 0 """
        broadcast_float_mask = \
            tf.expand_dims(
                tf.cast(mask, 'float32'),
                axis=-1
            )  # Add feature dim for broadcast, (None,n,1)

        """ Concatenate agents_feature and global_feature """
        x = inputs[0]  # (None,n,hidden_dim)
        x_global = inputs[1]  # (None, 4*hidden_dim)

        x_global = tf.expand_dims(x_global, axis=1)  # (None,1,4*hidden_dim)

        mult = tf.constant([1, self.config.max_num_blue_agents, 1])  # (1,n,1)
        x_global = tf.tile(x_global, mult)  # (None,n,4*hidden_dim)

        x = tf.concat([x, x_global], axis=-1)  # (None,n,5*hidden_dim)=(None,n,320)

        """ Q1 """
        x1 = self.dense1(x)  # (None,n,3*hidden_dim)=(None,n,192)
        # x1 = self.dropoout1(x1, training=training)
        x1 = self.dense2(x1)
        qs1 = self.dense3(x1)  # (None,n,action_dim)

        qs1 = qs1 * broadcast_float_mask  # (None,n,action_dim)

        """ Q2 """
        x10 = self.dense10(x)  # (None,n,3*hidden_dim)=(None,n,192)
        # x10 = self.dropoout10(x10, training=training)
        x10 = self.dense20(x10)
        qs10 = self.dense30(x10)  # (None,n,action_dim)

        qs10 = qs10 * broadcast_float_mask  # (None,n,action_dim)

        return qs1, qs10

    def build_graph(self, mask):
        x = tf.keras.layers.Input(
            shape=(self.config.max_num_red_agents,
                   self.config.hidden_dim)
        )  # (None,n,64)

        x_global = tf.keras.layers.Input(
            shape=(4 * self.config.hidden_dim,)
        )  # (None,256)

        model = tf.keras.models.Model(
            inputs=[x, x_global],
            outputs=self.call([x, x_global], mask),
            name='value_model'
        )

        return model


def main():
    from battlefield_strategy_pomdp_sp2 import BattleFieldStrategy  # used in main

    dir_name = './models_architecture'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    env = BattleFieldStrategy()
    env.reset()

    config = Config()

    grid_size = config.grid_size
    fov = config.fov
    com = config.com

    """ global_state & feature """
    global_ch = config.global_observation_channels  # 6
    global_n_frames = config.global_n_frames

    global_state_shape = (grid_size, grid_size, global_ch * global_n_frames)  # (15,15,6)

    global_state = np.ones(shape=global_state_shape)  # (15,15,6)
    global_state = np.expand_dims(global_state, axis=0)  # (1,15,15,6)

    global_cnn = GlobalCNNModel(config=config)
    global_feature = global_cnn(global_state)  # (1,hidden_dim)=(1,256)

    """ agent observation """
    ch = config.observation_channels
    n_frames = config.n_frames

    obs_shape = (2 * fov + 1, 2 * fov + 1, ch * n_frames)  # (5,5,16)
    pos_shape = (2 * n_frames,)  # (8,)

    max_num_agents = config.max_num_red_agents  # 15

    # Define alive_agents_ids & raw_obs
    alive_agents_ids = [0, 2, 3, 10]
    agent_obs = {}
    agent_pos = {}

    for i in alive_agents_ids:
        agent_id = 'red_' + str(i)
        agent_obs[agent_id] = np.ones(obs_shape)
        agent_pos[agent_id] = np.ones(pos_shape) * i  # (8,)

    # Get padded_obs, padded_pos
    padded_obs = \
        make_padded_obs(max_num_agents, obs_shape, agent_obs)  # (1,n,2*fov+1,2*fov+1,ch*n_frames)

    padded_pos = make_padded_pos(max_num_agents, pos_shape, agent_pos)  # (1,n,2*n_frames)

    # Get mask
    mask = make_mask(alive_agents_ids, max_num_agents)  # (1,n)

    # Get attention mask (adjacency matrix)
    float_mask = \
        tf.expand_dims(
            tf.cast(mask, 'float32'),
            axis=-1
        )  # (1,n,1)

    attention_mask = tf.matmul(
        float_mask, float_mask, transpose_b=True
    )  # (1,n,n)

    attention_mask = tf.cast(attention_mask, 'bool')

    """ cnn_model """
    cnn = CNNModel(config=config)

    features_cnn = cnn([padded_obs, padded_pos], mask)  # Build, (1,n,hidden_dim)

    cnn.summary()

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

    features_mha, score = mha(features_cnn,
                              mask,
                              attention_mask)  # Build, (None,n,hidden_dim),(1,num_heads,n,n)

    mha.summary()

    """ remove tf.function for summary """
    """
    idx = 1
    mha.build_graph(mask, attention_mask, idx).summary()

    tf.keras.utils.plot_model(
        mha.build_graph(mask, attention_mask, idx),
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

    policy_model.summary()

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

    q_model = SoftQModelGlobalState(config=config)

    qs1, qs2 = q_model([features_mha, global_feature], mask)
    print(f'Q1s.shape: {qs1.shape}, Q2s.shape: {qs2.shape}')

    q_model.summary()

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
