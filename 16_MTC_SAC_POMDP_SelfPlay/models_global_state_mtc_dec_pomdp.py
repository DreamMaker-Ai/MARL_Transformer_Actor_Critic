import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os

from config_dec_pomdp import Config
from sub_models_global_state_mtc_dec_pomdp import CNNModel, MultiHeadAttentionModel, PolicyModel, \
    SoftQModelGlobalState
from global_models_dec_pomdp import GlobalCNNModel
from utils_transformer_mtc_dec_pomdp import make_mask, make_padded_obs, make_padded_pos


class MarlTransformerGlobalStateModel(tf.keras.models.Model):
    """
    :params: n=max_num_agents
    :inputs: [[padded obs,padded_pos], global_obs]
             padded_obs: (None,n,2*fov+1,2*fov+1,ch*n_frames)
             padded_pos: (None,n,2*n_frames)
             global_obs: (None,g,g,global_ch*global_n_frames)
             (alive_)mask: (None,n)
             attention_mask: (None,n,n)

    :return: [policy_logits, [q1s, q2s]]:
                    [(None,n,action_dim), [(None,n,action_dim),(None,n,action_dim)]]
             [score1, score2]: [(None,num_heads,n,n),(None,num_heads,n,n)]

    Model: "marl_transformer_global_state_model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     global_cnn_model_1 (GlobalC  multiple                 258944
     NNModel)

     cnn_model (CNNModel)        multiple                  233120

     time_distributed_7 (TimeDis  multiple                 0
     tributed)

     multi_head_attention_model   multiple                 49792
     (MultiHeadAttentionModel)

     multi_head_attention_model_  multiple                 49792
     1 (MultiHeadAttentionModel)

     policy_model (PolicyModel)  multiple                  25157

     soft_q_model_global_state (  multiple                 148618
     SoftQModelGlobalState)

    =================================================================
    Total params: 765,423
    Trainable params: 765,423
    Non-trainable params: 0
    _________________________________________________________________
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
    def call(self, x, mask, attention_mask, training=True):
        """
         x=[[padded obs,padded_pos], global_obs]
             padded_obs: (None,n,2*fov+1,2*fov+1,ch*n_frames)=(None,15,5,5,4*4)
             padded_pos: (None,n,2*n_frames)=(None,15,8)
             global_obs: (None,g,g,global_ch*global_n_frames)=(None,15,15,6*1)
             (alive_)mask: (None,n)
             attention_mask: (None,n,n)
        """
        agents_obs = x[0][0]  # (None,15,5,5,16)
        agents_pos = x[0][1]  # (None,15,8)
        global_state = x[1]  # (None,15,15,6)

        """ Global feature by Global CNN layer """
        global_feature = self.global_cnn(global_state)  # (None,4*hidden_dim)=(None,256)

        """ CNN layer """
        features_cnn = self.cnn([agents_obs, agents_pos], mask)  # (None,n,hidden_dim)=(None,15,64)

        """ Dropout layer """
        features_cnn = self.dropout(features_cnn, training=training)

        """ Multi Head Self-Attention layer 1 """
        # features_mha1: (None,n,hidden_dim),
        # score1: (None,num_heads,n,n)
        features_mha1, score1 = self.mha1(features_cnn, mask, attention_mask, training=training)

        """ Multi Head Self-Attention layer 2 """
        # features_mha2: (None,n,hidden_dim),
        # score2: (None,num_heads,n,n)
        features_mha2, score2 = self.mha2(features_mha1, mask, attention_mask, training=training)

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

    def sample_actions(self, x, mask, attention_mask, training=False):
        # Use only agents obs & pos.
        # x=[padded obs, padded_pos]
        #   padded_obs: (None,n,2*fov+1,2*fov+1,ch*n_frames)=(None,15,5,5,16)
        #   padded_pos: (None,n,2*n_frames)=(None,15,8)
        # mask: (b,n)
        # attention_mask: (b,n,n)
        """ action=5 if policyprobs=[0,0,0,0,0], that is or the dead or dummy agents """

        # [policy_logits, _], scores = self(states, mask, training=training)
        features_cnn = self.cnn(x, mask)

        features_cnn = self.dropout(features_cnn, training=True)

        features_mha1, score1 = self.mha1(features_cnn, mask, attention_mask, training=True)
        features_mha2, score2 = self.mha2(features_mha1, mask, attention_mask, training=True)

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
        x0 = tf.keras.layers.Input(
            shape=(self.config.max_num_red_agents,
                   2 * self.config.fov + 1,
                   2 * self.config.fov + 1,
                   self.config.observation_channels * self.config.n_frames)
        )
        x1 = tf.keras.layers.Input(
            shape=(self.config.max_num_red_agents,
                   2 * self.config.n_frames)
        )

        x = [x0, x1]

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
    from battlefield_strategy_team_reward_global_state import BattleFieldStrategy  # used in main

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

    """ Make model """
    marl_transformer = MarlTransformerGlobalStateModel(config=config)

    [policy_logits, [q1s, q2s]], scores = \
        marl_transformer([[padded_obs, padded_pos], global_state],
                         mask, attention_mask, training=True)

    marl_transformer.summary()

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
    actions, _ = \
        marl_transformer.sample_actions([padded_obs, padded_pos],
                                        mask, attention_mask, training=False)  # (b,n), int32

    print('\n')
    print(probs)
    print(actions)


if __name__ == '__main__':
    main()
