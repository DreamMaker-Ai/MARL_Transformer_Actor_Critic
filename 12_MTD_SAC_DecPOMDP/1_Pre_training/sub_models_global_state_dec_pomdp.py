import os.path

import keras.layers
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

from config_dec_pomdp import Config
from battlefield_strategy_team_reward_global_state import BattleFieldStrategy
from utils_transformer_dec_pomdp import make_po_id_mask as make_mask
from utils_transformer_dec_pomdp import make_padded_obs
from global_models_dec_pomdp import GlobalCNNModel


class CNNModel(tf.keras.models.Model):
    """
    :inputs: [obs,pos]=[(b,5,5,16),(b,8)]
    :return: (None,hidden_dim)=(None,64)

    Model: "cnn_model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     conv2d_4 (Conv2D)           multiple                  1088

     conv2d_5 (Conv2D)           multiple                  73856

     conv2d_6 (Conv2D)           multiple                  147584

     flatten_1 (Flatten)         multiple                  0

     dense_1 (Dense)             multiple                  288

     concatenate (Concatenate)   multiple                  0

     dense_2 (Dense)             multiple                  10304

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
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=1,
                strides=1,
                activation='relu',
                kernel_initializer='Orthogonal'
            )

        self.conv1 = \
            tf.keras.layers.Conv2D(
                filters=128,
                kernel_size=3,
                strides=1,
                activation='relu',
                kernel_initializer='Orthogonal'
            )

        self.conv2 = \
            tf.keras.layers.Conv2D(
                filters=128,
                kernel_size=3,
                strides=1,
                activation='relu',
                kernel_initializer='Orthogonal'
            )

        self.flatten1 = tf.keras.layers.Flatten()

        self.dense_pos_enc = \
            tf.keras.layers.Dense(
                units=32,
                activation='relu',
            )

        self.concatenate = tf.keras.layers.Concatenate(axis=-1)

        self.dense1 = \
            tf.keras.layers.Dense(
                units=self.config.hidden_dim,
                activation=None
            )

    @tf.function
    def call(self, inputs):
        # inputs: [obs, pos]
        #   :obs: (b,2*fov+1,2*fov+1,ch*n_frames)=(1,5,5,16)
        #   :pos: (b,2*n_frames)=(1,8)

        h = self.conv0(inputs[0])  # (1,5,5,64)
        h = self.conv1(h)  # (1,3,3,128)
        h = self.conv2(h)  # (1,1,1,128)

        h1 = self.flatten1(h)  # (1,128)

        pos_enc = self.dense_pos_enc(inputs[1])  # (1,32)

        z = self.concatenate([h1, pos_enc])  # (1,160)

        features = self.dense1(z)  # (1,64)

        return features

    def build_graph(self):
        """ For summary & plot_model """
        x0 = tf.keras.layers.Input(
            shape=(2 * self.config.fov + 1,
                   2 * self.config.fov + 1,
                   self.config.observation_channels * self.config.n_frames)
        )

        x1 = tf.keras.layers.Input(
            shape=(2 * self.config.n_frames,)
        )

        x = [x0, x1]  # [obs,pos]

        model = \
            tf.keras.models.Model(
                inputs=[x],
                outputs=self.call(x),
                name='cnn_model'
            )

        return model


class MultiHeadAttentionModel(tf.keras.models.Model):
    """
    Two layers of MultiHeadAttention (Self Attention with provided mask)

    :param mask: (None,1,n), bool
    :param max_num_agents=15=n
    :param hidden_dim = 64

    :return: features: (None,hidden_dim)=(None,64)
             score: (None,num_heads,n)=(None,2,15)

    Model: "multi_head_attention_model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     lambda (Lambda)             multiple                  0

     lambda_1 (Lambda)           multiple                  0

     multi_head_attention (Multi  multiple                 33216
     HeadAttention)

     lambda_2 (Lambda)           multiple                  0

     add (Add)                   multiple                  0

     dense_3 (Dense)             multiple                  8320

     dense_4 (Dense)             multiple                  8256

     dropout (Dropout)           multiple                  0

     add_1 (Add)                 multiple                  0

     reshape (Reshape)           multiple                  0 (unused)

    =================================================================
    Total params: 49,792
    Trainable params: 49,792
    Non-trainable params: 0
    _________________________________________________________________
    """

    def __init__(self, config, **kwargs):
        super(MultiHeadAttentionModel, self).__init__(**kwargs)

        self.config = config

        self.query_feature = \
            tf.keras.layers.Lambda(
                lambda x: tf.cast(tf.expand_dims(x, axis=1), dtype=tf.float32)
            )

        self.features = \
            tf.keras.layers.Lambda(
                lambda x: tf.cast(tf.stack(x, axis=1), dtype=tf.float32)
            )

        self.mha1 = \
            tf.keras.layers.MultiHeadAttention(
                num_heads=self.config.num_heads,
                key_dim=self.config.key_dim,
            )

        self.squeeze1 = tf.keras.layers.Lambda(
            lambda x: tf.cast(tf.squeeze(x, axis=1), dtype=tf.float32)
        )

        self.add1 = \
            tf.keras.layers.Add()

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

        self.reshape = tf.keras.layers.Reshape(target_shape=(config.hidden_dim,))

    @tf.function
    def call(self, inputs, mask=None, training=True):
        # inputs: [(None,hiddendim),[(None,hidden_dim),...,(None,hidden_dim)]]
        #           =[(1,64),[(1,64),...,(1,64)]]
        # mask: (None,1,n)=(1,1,15), bool,  n=15: max_num_agents

        attention_mask = tf.cast(mask, 'bool')  # (None,1,n)=(1,1,15)

        query_feature = self.query_feature(inputs[0])  # (None,1,hidden_dim)=(1,1,64)
        features = self.features(inputs[1])  # (None,n,hidden_dim)=(1,15,64)

        x, score = \
            self.mha1(
                query=query_feature,
                key=features,
                value=features,
                attention_mask=attention_mask,
                return_attention_scores=True,
            )  # (None,1,hidden_dim),(None,num_heads,1,n)=(1,1,64),(1,2,1,15)

        x = self.squeeze1(x)  # (None,hidden_dim)=(1,64)

        x1 = self.add1([inputs[0], x])  # (None,hidden_dim)=(1,64)

        x2 = self.dense1(x1)  # (None,hidden_dim*2)=(1,128)

        x2 = self.dense2(x2)  # (None,n,hidden_dim)=(1,64)

        x2 = self.dropoout1(x2, training=training)

        feature = self.add2([x1, x2])  # (None,hidden_dim)=(1,64)

        score = tf.squeeze(score, axis=2)  # (1,2,15)

        return feature, score  # (None,hidden_dim), (None,num_heads,n)

    def build_graph(self, mask, idx):
        query = tf.keras.layers.Input(shape=(self.config.hidden_dim,))
        features = [tf.keras.layers.Input(shape=(self.config.hidden_dim,))
                    for _ in range(self.config.max_num_red_agents)]

        x = [query, features]

        model = tf.keras.models.Model(
            inputs=x,
            outputs=self.call(x, mask, training=True),
            name='mha_' + str(idx),
        )

        return model


class PolicyModel(tf.keras.models.Model):
    """
    :param action_dim=5
    :param hidden_dim=64
    :return: Policy logits, (None,action_dim)=(None,5)

    Model: "policy_model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     dense_5 (Dense)             multiple                  12480

     dense_6 (Dense)             multiple                  12352

     dense_7 (Dense)             multiple                  325

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
            tf.keras.layers.Dense(
                units=self.config.hidden_dim * 3,
                activation='relu',
            )

        # self.dropoout1 = tf.keras.layers.Dropout(rate=self.config.dropout_rate)

        self.dense2 = \
            tf.keras.layers.Dense(
                units=self.config.hidden_dim,
                activation='relu',
            )

        self.dense3 = \
            tf.keras.layers.Dense(
                units=self.config.action_dim,
                activation=None,
            )

    @tf.function
    def call(self, inputs, training=True):
        # inputs: (None,hidden_dim)=(None,64)
        # prob_logit=0 for dead / dummy agents

        x1 = self.dense1(inputs)  # (None,hidden_dim*3)=(1,192)

        # x1 = self.dropoout1(x1, training=training)

        x1 = self.dense2(x1)  # (None,hidden_dim)=(1,64)

        logit = self.dense3(x1)  # (None,action_dim)=(1,5)

        return logit

    def build_graph(self, ):
        x = tf.keras.layers.Input(
            shape=(self.config.hidden_dim,))  # (None,64)

        model = tf.keras.models.Model(
            inputs=[x],
            outputs=self.call(x),
            name='policy_logits'
        )

        return model


class SoftQModelGlobalState(tf.keras.models.Model):
    """
    Use double Q trick

    :param action_dim=5
    :param hidden_dim=64
    :inputs: [agents_feature, global_feature]=[(None,hidden_dim),(None,hidden_dim*4)]
    :return: soft_Q1, soft_Q2; (None,action_dim), (None,action_dim)

    Model: "soft_q_model_global_state"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     dense_8 (Dense)             multiple                  61632

     dense_9 (Dense)             multiple                  12352

     dense_10 (Dense)            multiple                  325

     dense_11 (Dense)            multiple                  61632

     dense_12 (Dense)            multiple                  12352

     dense_13 (Dense)            multiple                  325

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
            tf.keras.layers.Dense(
                units=self.config.hidden_dim * 3,
                activation='relu',
            )

        # self.dropoout1 = tf.keras.layers.Dropout(rate=self.config.dropout_rate)

        self.dense2 = \
            tf.keras.layers.Dense(
                units=self.config.hidden_dim,
                activation='relu',
            )

        self.dense3 = \
            tf.keras.layers.Dense(
                units=self.action_dim,
                activation=None,
            )

        # For Q2
        self.dense10 = \
            tf.keras.layers.Dense(
                units=self.config.hidden_dim * 3,
                activation='relu',
            )

        # self.dropoout10 = tf.keras.layers.Dropout(rate=self.config.dropout_rate)

        self.dense20 = \
            tf.keras.layers.Dense(
                units=self.config.hidden_dim,
                activation='relu',
            )

        self.dense30 = \
            tf.keras.layers.Dense(
                units=self.action_dim,
                activation=None,
            )

    @tf.function
    def call(self, inputs, training=True):
        """
        Q1, Q2 = 0 for dead / dummy agents
        inputs=[agent_feature, global_feature]=[(None,hidden_dim),(None,hidden_dim*4)]
        :return: Q1, Q2; (None,action_dim), (None,action_dim)
        """

        """ Concatenate agents_feature and global_feature """
        x = inputs[0]  # (None,hidden_dim)=(1,64)
        x_global = inputs[1]  # (None, hidden_dim*4)=(1,256)

        x = tf.concat([x, x_global], axis=-1)  # (None,5*hidden_dim)=(None,320)

        """ Q1 """
        x1 = self.dense1(x)  # (None,n,3*hidden_dim)=(None,192)
        # x1 = self.dropoout1(x1, training=training)
        x1 = self.dense2(x1)  # (None,hidden_dim)=(1,64)
        qs1 = self.dense3(x1)  # (None,action_dim)

        """ Q2 """
        x10 = self.dense10(x)  # (None,3*hidden_dim)=(None,768)
        # x10 = self.dropoout10(x10, training=training)
        x10 = self.dense20(x10)
        qs10 = self.dense30(x10)  # (None,action_dim)

        return qs1, qs10

    def build_graph(self, ):
        x = tf.keras.layers.Input(shape=(self.config.hidden_dim,))  # (None,64)

        x_global = tf.keras.layers.Input(
            shape=(self.config.hidden_dim * 4,)
        )  # (None,256)

        model = tf.keras.models.Model(
            inputs=[x_global, x],
            outputs=self.call([x, x_global]),
            name='value_model'
        )

        return model


def main():
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
    global_n_frames = config.global_n_frames  # 1

    global_state_shape = (grid_size, grid_size, global_ch * global_n_frames)  # (15,15,6)

    global_state = np.ones(shape=global_state_shape)  # (15,15,6)
    global_state = np.expand_dims(global_state, axis=0)  # (1,15,15,6)

    global_cnn = GlobalCNNModel(config=config)
    global_feature = global_cnn(global_state)  # (1,hidden_dim*4)=(1,256)

    """ agent observation """
    ch = config.observation_channels  # 4
    n_frames = config.n_frames  # 4

    obs_shape = (2 * fov + 1, 2 * fov + 1, ch * n_frames)  # (5,5,16)
    max_num_agents = config.max_num_red_agents  # 15

    # Define alive_agents_ids & raw_obs
    alive_agents_ids = [0, 2]

    """ mask """
    masks = make_mask(alive_agents_ids, max_num_agents, env.reds, com)
    # [(1,1,n),...], len=n

    """ cnn_model """
    cnn = CNNModel(config=config)

    # Get features list of all agents
    features = []
    for i in range(max_num_agents):
        if i in alive_agents_ids:
            obs = np.random.rand(obs_shape[0], obs_shape[1], obs_shape[2]).astype(np.float32)
            pos = np.random.rand(2 * n_frames).astype(np.float32)
        else:
            obs = np.zeros((obs_shape[0], obs_shape[1], obs_shape[2])).astype(np.float32)
            pos = np.zeros((2 * n_frames,)).astype(np.float32)

        obs = np.expand_dims(obs, axis=0)  # (1,5,5,16)
        pos = np.expand_dims(pos, axis=0)  # (1,8)
        feat = cnn([obs, pos])  # (1,64)
        features.append(feat)  # [(1,64),...,(1,64)], len=15

    cnn.summary()
    print('\n')

    """ remove tf.function for summary """
    """
    cnn.build_graph().summary()

    tf.keras.utils.plot_model(
        cnn.build_graph(),
        to_file=dir_name + '/cnn_model',
        show_shapes=True,
        show_layer_activations=True,
        show_dtype=True,
        dpi=96 * 3
    )
    """

    """ mha model """
    mha = MultiHeadAttentionModel(config=config)

    # Get output list of attention of all agents
    att_features = []
    att_scores = []
    for i in range(max_num_agents):
        query_feature = features[i]  # (1,64)

        inputs = [query_feature, features]  # [(1,64),[(1,64),...,(1,64)]]

        att_feature, att_score = \
            mha(inputs,
                masks[i],
                training=True
                )  # (None,hidden_dim),(None,num_heads,n)

        att_features.append(att_feature)  # [(1,64),...], len=n=15
        att_scores.append(att_score)  # [(1,2,15),...], len=15

    mha.summary()
    print('\n')

    """ remove tf.function for summary """
    """
    agent_id = 1
    idx = 1
    mha.build_graph(masks[agent_id], idx).summary()

    tf.keras.utils.plot_model(
        mha.build_graph(masks[agent_id], idx),
        to_file=dir_name + '/mha_model_' + str(idx),
        show_shapes=True,
        show_layer_activations=True,
        show_dtype=True,
        dpi=96 * 3
    )
    """

    """ policy model """
    policy = PolicyModel(config=config)

    # Get policy_logits list of all agents
    policy_logits = []
    for i in range(max_num_agents):
        policy_logit = policy(att_features[i])  # (None,5)
        policy_logits.append(policy_logit)  # [(None,5),...], len=15

    policy.summary()
    print('\n')

    """ remove tf.function for summary """
    """
    policy.build_graph().summary()

    tf.keras.utils.plot_model(
        policy.build_graph(),
        to_file=dir_name + '/policy_model',
        show_shapes=True,
        show_layer_activations=True,
        show_dtype=True,
        dpi=96 * 3
    )
    """

    """ q_model """
    q_net = SoftQModelGlobalState(config=config)

    # Get q_logits (of Q1 & Q2) list of all agents
    qs = []
    for i in range(max_num_agents):
        q = q_net([att_features[i], global_feature])  # double Q trick [(None,5),(None,5)]
        qs.append(q)  # [[(None,5),(None,5)],...], len=n=15

    q_net.summary()
    print('\n')

    """ remove tf.function for summary """
    """
    q_net.build_graph().summary()

    tf.keras.utils.plot_model(
        q_net.build_graph(),
        to_file=dir_name + '/q_model',
        show_shapes=True,
        show_layer_activations=True,
        show_dtype=True,
        dpi=96 * 3
    )
    """


if __name__ == '__main__':
    main()
