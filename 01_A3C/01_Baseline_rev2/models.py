import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os

from config import Config
from sub_models import CNNModel, MultiHeadAttentionModel, PolicyModel, ValueModel
from utils_transformer import make_mask, make_padded_obs


class MarlTransformerModel(tf.keras.models.Model):
    """
    :inputs: padded obs (None,n,g,g,ch*n_frames), n=max_num_agents
             mask (None,n,n)
    :return: [policy_probs, values]: [(None,n,action_dim), (None,n,1)]
             [score1, score2]: [(None,num_heads,n,n),(None,num_heads,n,n)]

    Model: "marl_transformer"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to
    ==================================================================================================
     input_1 (InputLayer)           [(None, 15, 15, 15,  0           []
                                     6)]

     cnn_model (CNNModel)           (None, 15, 256)      258944      ['input_1[0][0]']

     dropout (Dropout)              (None, 15, 256)      0           ['cnn_model[0][0]']

     multi_head_attention_model (Mu  ((None, 15, 256),   526080      ['dropout[0][0]']
     ltiHeadAttentionModel)          (None, 2, 15, 15))

     multi_head_attention_model_1 (  ((None, 15, 256),   526080      ['multi_head_attention_model[0][0
     MultiHeadAttentionModel)        (None, 2, 15, 15))              ]']

     policy_model (PolicyModel)     (None, 15, 5)        395525      ['multi_head_attention_model_1[0]
                                                                     [0]']

     value_model (ValueModel)       (None, 15, 1)        394497      ['multi_head_attention_model_1[0]
                                                                     [0]']

    ==================================================================================================
    Total params: 2,101,126
    Trainable params: 2,101,126
    Non-trainable params: 0
    __________________________________________________________________________________________________
    """

    def __init__(self, config, **kwargs):
        super(MarlTransformerModel, self).__init__(**kwargs)

        self.config = config

        """ Prepare sub models """
        self.cnn = CNNModel(config=self.config)

        self.dropout = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dropout(rate=self.config.dropout_rate)
        )

        self.mha1 = MultiHeadAttentionModel(config=self.config)

        self.mha2 = MultiHeadAttentionModel(config=self.config)

        self.policy = PolicyModel(config=self.config)
        self.value = ValueModel(config=self.config)

    @tf.function
    def call(self, x, mask, training=True):
        # x: (None,n,g,g,ch*n_frames)=(None,17,20,20,16), mask:(None,n,n)=(none,17,17)

        """ CNN layer """
        features_cnn = self.cnn(x, mask)  # (1,n,hidden_dim)

        """ Dropout layer """
        features_cnn = self.dropout(features_cnn, training=training)

        """ Multi Head Self-Attention layer 1 """
        # features_mha1: (None,n,hidden_dim),
        # score1: (None,num_heads,n,n)
        features_mha1, score1 = self.mha1(features_cnn, mask, training=training)

        """ Multi Head Self-Attention layer 2 """
        # features_mha2: (None,n,hidden_dim),
        # score2: (None,num_heads,n,n)
        features_mha2, score2 = self.mha2(features_mha1, mask, training=training)

        """ Policy (policy_probs output) """
        policy_probs = self.policy(features_mha2, mask, training=training)  # (None,n,action_dim)

        """ Values """
        values = self.value(features_mha2, mask, training=training)  # (None,n,1)

        return [policy_probs, values], [score1, score2]

    def sample_actions(self, states, mask, training=False):
        # states : (b,n,g,g,ch*n_frames)
        # mask: (b,n)
        """ action=5 if policyprobs=[0,0,0,0,0], that is or the dead or dummy agents """

        [policy_probs, values], scores = self(states, mask, training=training)

        num_agents = self.config.max_num_red_agents
        actions = []

        for i in range(num_agents):
            cdist = tfp.distributions.Categorical(probs=policy_probs[:, i, :])
            action = cdist.sample()  # (b,)
            actions.append(np.expand_dims(action.numpy(), axis=-1))

        actions = np.concatenate(actions, axis=-1)  # (b,n)

        return actions  # action=5 for the dead or dummy agents

    def build_graph(self, mask):
        x = tf.keras.layers.Input(
            shape=(self.config.max_num_red_agents,
                   self.config.grid_size,
                   self.config.grid_size,
                   self.config.observation_channels * self.config.n_frames)
        )

        features_cnn = self.cnn(x, mask)

        features_cnn = self.dropout(features_cnn, training=True)

        features_mha1, score1 = self.mha1(features_cnn, mask, training=True)
        features_mha2, score2 = self.mha2(features_mha1, mask, training=True)

        policy_probs = self.policy(features_mha2, mask)
        values = self.value(features_mha2, mask)

        model = tf.keras.models.Model(
            inputs=[x],
            outputs=[[policy_probs, values], [score1, score2]],
            name='marl_transformer',
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

    """ Make model """
    marl_transformer = MarlTransformerModel(config=config)

    [policy_probs, values], scores = marl_transformer(padded_obs, mask, training=True)

    """ Summary """
    print('\n')
    print('--------------------------------------- model ---------------------------------------')
    marl_transformer.build_graph(mask).summary()

    tf.keras.utils.plot_model(
        marl_transformer.build_graph(mask),
        to_file=dir_name + '/marl_transformer',
        show_shapes=True,
        show_layer_activations=True,
        show_dtype=True,
        dpi=96 * 3
    )

    """ Sample actions """
    actions = marl_transformer.sample_actions(padded_obs, mask, training=False)  # (b,n), int32

    print('\n')
    print(policy_probs)
    print(actions)


if __name__ == '__main__':
    main()
