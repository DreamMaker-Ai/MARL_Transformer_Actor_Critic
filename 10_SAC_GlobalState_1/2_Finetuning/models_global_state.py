import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os

from config_finetuning import Config
from sub_models_global_state import CNNModel, MultiHeadAttentionModel, PolicyModel, \
    SoftQModelGlobalState
from global_models import GlobalCNNModel
from utils_transformer import make_mask, make_padded_obs


class MarlTransformerGlobalStateModel(tf.keras.models.Model):
    """
    :inputs: [padded obs, global_obs]=
             [(None,n,g,g,ch*n_frames),(None,g,g,global_ch*global_n_frames)],
             mask (None,n,n),  n=max_num_agents
    :return: [policy_logits, [q1s, q2s]]:
                    [(None,n,action_dim), [(None,n,action_dim),(None,n,action_dim)]]
             [score1, score2]: [(None,num_heads,n,n),(None,num_heads,n,n)]

    Model: "marl_transformer"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to
    ==================================================================================================
     input_1 (InputLayer)           [(None, 15, 15, 15,  0           []
                                     6)]

     cnn_model (CNNModel)           (None, 15, 256)      258944      ['input_1[0][0]']

     time_distributed_6 (TimeDistri  (None, 15, 256)     0           ['cnn_model[0][0]']
     buted)

     multi_head_attention_model (Mu  ((None, 15, 256),   526080      ['time_distributed_6[0][0]']
     ltiHeadAttentionModel)          (None, 2, 15, 15))

     input_2 (InputLayer)           [(None, 15, 15, 6)]  0           []

     multi_head_attention_model_1 (  ((None, 15, 256),   526080      ['multi_head_attention_model[0][0
     MultiHeadAttentionModel)        (None, 2, 15, 15))              ]']

     global_cnn_model (GlobalCNNMod  (None, 256)         258944      ['input_2[0][0]']
     el)

     policy_model (PolicyModel)     (None, 15, 5)        395525      ['multi_head_attention_model_1[0]
                                                                     [0]']

     soft_q_model_global_state (Sof  ((None, 15, 5),     1184266     ['multi_head_attention_model_1[0]
     tQModelGlobalState)             (None, 15, 5))                  [0]',
                                                                      'global_cnn_model[0][0]']

    ==================================================================================================
    Total params: 3,149,839
    Trainable params: 3,149,839
    Non-trainable params: 0
    __________________________________________________________________________________________________
    """

    def __init__(self, config, **kwargs):
        super(MarlTransformerGlobalStateModel, self).__init__(**kwargs)

        self.config = config

        """ Prepare sub models """
        self.global_cnn = GlobalCNNModel(config=self.config)

        self.cnn = CNNModel(config=self.config)

        self.dropout = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dropout(rate=self.config.dropout_rate)
        )

        self.mha1 = MultiHeadAttentionModel(config=self.config)

        self.mha2 = MultiHeadAttentionModel(config=self.config)

        self.policy = PolicyModel(config=self.config)
        self.value = SoftQModelGlobalState(config=self.config)

    @tf.function
    def call(self, x, mask, training=True):
        """
        x=[agents_obs, global_state]
         =[(None,n,g,g,ch*n_frames),(None,g,g,global_ch*global_n_frames)]
         =[(None,15,15,15,6*1),(None,15,15,6*1)],
        mask:(None,n)=(None,15)
        """
        agents_obs = x[0]  # (None,15,15,15,6)
        global_state = x[1]  # (None,15,15,6)

        """ Global feature by Global CNN layer """
        global_feature = self.global_cnn(global_state)  # (None,hidden_dim)

        """ CNN layer """
        features_cnn = self.cnn(agents_obs, mask)  # (None,n,hidden_dim)

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
        policy_logits = self.policy(features_mha2, mask, training=training)  # (None,n,action_dim)

        """ [q1s, q2s] """
        [q1s, q2s] = self.value([features_mha2, global_feature], mask, training=training)
        # [(None,n,action_dim),(None,n,action_dim)]]

        return [policy_logits, [q1s, q2s]], [score1, score2]

    @staticmethod
    def process_action(action_logits, mask):
        """ probs=0, log_probs=0 of dead / dummy agents """

        broadcast_float_mask = tf.expand_dims(tf.cast(mask, 'float32'), axis=-1)  # (None,n,1)

        probs = tf.nn.softmax(action_logits)  # (None,n,action_dim)
        probs = probs * broadcast_float_mask  # (None,n,action_dim)

        log_probs = tf.nn.log_softmax(action_logits)  # (None,n,action_dim)
        log_probs = log_probs * broadcast_float_mask  # (None,n,action_dim)

        return probs, log_probs

    def sample_actions(self, x, mask, training=False):
        # Use only agents obs.
        # x : agents_obs, # (b,n,g,g,ch*n_frames)
        # mask: # (b,n)
        """ action=5 if policyprobs=[0,0,0,0,0], that is or the dead or dummy agents """

        # [policy_logits, _], scores = self(states, mask, training=training)
        features_cnn = self.cnn(x, mask)

        features_cnn = self.dropout(features_cnn, training=True)

        features_mha1, score1 = self.mha1(features_cnn, mask, training=True)
        features_mha2, score2 = self.mha2(features_mha1, mask, training=True)

        policy_logits = self.policy(features_mha2, mask)
        scores = [score1, score2]

        policy_probs, _ = self.process_action(policy_logits, mask)

        num_agents = self.config.max_num_red_agents
        actions = []

        for i in range(num_agents):
            cdist = tfp.distributions.Categorical(probs=policy_probs[:, i, :])
            action = cdist.sample()  # (b,)
            actions.append(np.expand_dims(action.numpy(), axis=-1))

        actions = np.concatenate(actions, axis=-1)  # (b,n)

        return actions, scores  # action=5 for the dead or dummy agents, [score1, score2]

    def build_graph(self, mask):
        x = tf.keras.layers.Input(
            shape=(self.config.max_num_red_agents,
                   self.config.grid_size,
                   self.config.grid_size,
                   self.config.observation_channels * self.config.n_frames)
        )

        global_x = tf.keras.layers.Input(
            shape=(self.config.grid_size,
                   self.config.grid_size,
                   self.config.global_observation_channels * self.config.global_n_frames)
        )

        global_feature = self.global_cnn(global_x)

        features_cnn = self.cnn(x, mask)

        features_cnn = self.dropout(features_cnn, training=True)

        features_mha1, score1 = self.mha1(features_cnn, mask, training=True)
        features_mha2, score2 = self.mha2(features_mha1, mask, training=True)

        policy_logits = self.policy(features_mha2, mask)
        [q1s, q2s] = self.value([features_mha2, global_feature], mask)

        model = tf.keras.models.Model(
            inputs=[x, global_x],
            outputs=[[policy_logits, [q1s, q2s]], [score1, score2]],
            name='marl_transformer',
        )

        return model


def main():
    dir_name = './models_architecture'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    config = Config()

    grid_size = config.grid_size

    """ global_state """
    global_ch = config.global_observation_channels  # 6
    global_n_frames = config.global_n_frames

    global_state_shape = (grid_size, grid_size, global_ch * global_n_frames)  # (15,15,6)

    global_state = np.ones(shape=global_state_shape)  # (15,15,6)
    global_state = np.expand_dims(global_state, axis=0)  # (1,15,15,6)

    """ agents obs """
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
    marl_transformer = MarlTransformerGlobalStateModel(config=config)

    [policy_logits, [q1s, q2s]], scores = \
        marl_transformer([padded_obs, global_state], mask, training=True)

    """ Summary """
    """
    print('\n')
    print('--------------------------------------- model ---------------------------------------')
    marl_transformer.build_graph(mask).summary()

    tf.keras.utils.plot_model(
        marl_transformer.build_graph(mask),
        to_file=dir_name + '/marl_transformer_with_global_state',
        show_shapes=True,
        show_layer_activations=True,
        show_dtype=True,
        dpi=96 * 3
    )
    """

    probs, log_probs = marl_transformer.process_action(policy_logits, mask)
    print(f"probs: {probs}, {probs.shape}")
    print(f"log_probs: {log_probs}, {log_probs.shape}")

    """ Sample actions """
    actions, _ = marl_transformer.sample_actions(padded_obs, mask, training=False)  # (b,n), int32

    print('\n')
    print(probs)
    print(actions)


if __name__ == '__main__':
    main()
