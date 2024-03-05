import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os

from config import Config
from sub_models_global_state import CNNModel, MultiHeadAttentionModel, PolicyModel, \
    SoftQModelGlobalState
from global_models import GlobalCNNModel
from utils_transformer import make_id_mask as make_mask


class MarlTransformerGlobalStateModel(tf.keras.models.Model):
    """
    n=max_num_agents=15
    :inputs: [agents_obs, global_obs], masks
                :agents_obs: [(None,g,g,ch*n_frames),...], len=n
                :global_obs: (None,g,g,global_ch*global_n_frames),
                :masks: [(None,1,n),...], len=n

    :return: policy_logits, q1s, q2s scores1, scores2:
                :policy_logits: [(None,action_dim),...], len=n
                :q1s, q2s: [(None,action_dim),...], len=n
                :scores1, scores2: (None,num_heads,n),...], len=n

    Model: "marl_transformer_global_state_model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     global_cnn_model (GlobalCNN  multiple                 258944
     Model)

     cnn_model (CNNModel)        multiple                  258944

     dropout (Dropout)           multiple                  0

     multi_head_attention_model   multiple                 526080
     (MultiHeadAttentionModel)

     multi_head_attention_model_  multiple                 526080
     1 (MultiHeadAttentionModel)

     policy_model (PolicyModel)  multiple                  395525

     soft_q_model_global_state (  multiple                 1184266
     SoftQModelGlobalState)

    =================================================================
    Total params: 3,149,839
    Trainable params: 3,149,839
    Non-trainable params: 0
    _________________________________________________________________
    """

    def __init__(self, config, **kwargs):
        super(MarlTransformerGlobalStateModel, self).__init__(**kwargs)

        self.config = config

        """ Prepare sub models """
        self.global_cnn = GlobalCNNModel(config=self.config)

        self.cnn = CNNModel(config=self.config)

        self.dropout = tf.keras.layers.Dropout(rate=self.config.dropout_rate)

        self.mha1 = MultiHeadAttentionModel(config=self.config)

        self.mha2 = MultiHeadAttentionModel(config=self.config)

        self.policy = PolicyModel(config=self.config)
        self.value = SoftQModelGlobalState(config=self.config)

    def cnn_block(self, agents_obs, masks, training=False):
        """
        :agents_obs: [(None,g,g,ch*n_frames),...], len=n
        :masks: [(None,1,n),...], len=n

        :return: features_cnn: [(None,hidden_dim),...], len=n
        """
        features_cnn = []
        for i in range(self.config.max_num_red_agents):
            feature = self.cnn(agents_obs[i])  # (None,hidden_dim)

            mask = masks[i]  # (None,1,n)
            float_mask = tf.cast(mask[:, :, i], 'float32')
            # (None,1); agent_i alive=1, dead/dummy=0

            feature = feature * float_mask  # (None,hidden_dim)  # feature=0 for dead / dummy agents

            features_cnn.append(feature)  # [(None,hidden_dim),...], len=n

        return features_cnn

    def dropout_block(self, features_cnn, training=False):
        """
        features_cnn: [(None,hidden_dim),...], len=n
        :return: features_dropout: [(None,hidden_dim),...], len=n
        """
        features_dropout = []

        for feature in features_cnn:
            feature = self.dropout(feature, training)  # (None,hidden_dim)

            features_dropout.append(feature)  # [(None,hidden_dim),...], len=n

        return features_dropout

    def mha_block(self, mha, features, masks, training=False):
        """
        :param features: [(None,hidden_dim),...], len=n
        :param masks: [(None,1,n),...], len=n
        :return: att_features: [(None,hidden_dim),...], len=n
               : att_scores: [(None,num_heads,n),...], len=n
        """
        att_features = []
        att_scores = []

        for i in range(self.config.max_num_red_agents):
            query_feature = features[i]  # (None,hidden_dim)
            inputs = [query_feature, features]  # [(None,hidden_dim),[(None,hidden_dim),...]]

            mask = masks[i]  # (None,1,n)

            att_feature, att_score = mha(inputs, mask, training)
            # (None,hidden_dim), (None,num_heads,n)

            float_mask = tf.cast(mask[:, :, i], 'float32')  # (None,1), alive=1, dead/dummy=0
            att_feature = att_feature * float_mask  # (None,hidden_dim)

            float_mask = tf.cast(tf.expand_dims(mask[:, :, i], axis=1), 'float32')
            # (None,1,1), add head_dim
            att_score = att_score * float_mask  # (None,num_heads,n)

            att_features.append(att_feature)  # [(None,hidden_dim),...], len=n
            att_scores.append(att_score)  # [(None,num_heads,n),...], len=n

        return att_features, att_scores

    def policy_block(self, features_mha, masks, training=False):
        """
        :param features_mha:  [(None,hidden_dim),...], len=n
        :param masks: [(None,1,n),...], len=n
        :return: policy_logits: [(None,action_dim),...], len=n
        """
        policy_logits = []

        for i in range(self.config.max_num_red_agents):
            policy_logit = self.policy(features_mha[i])  # (None,action_dim)

            mask = masks[i]  # (None,1,n)
            float_mask = tf.cast(mask[:, :, i], 'float32')  # (None,1), alive=1, dead/dummy=0
            policy_logit = policy_logit * float_mask  # (None,action_dim)

            policy_logits.append(policy_logit)  # [(None,action_dim),...], len=n

        return policy_logits  # [(None,action_dim),...], len=n

    def value_block(self, features_mha, global_feature, masks, training=False):
        """
        :param features_mha:  [(None,hidden_dim),...], len=n
        :param global_feature:  (None,hidden_dim)
        :param masks: [(None,1,n),...], len=n
        :return: Q1, Q2 (Double Q Trick): [(None,action_dim),...]
        """
        q1s = []
        q2s = []

        for i in range(self.config.max_num_red_agents):
            q1, q2 = self.value([features_mha[i], global_feature])  # (None,action_dim)

            mask = masks[i]  # (None,1,n)
            float_mask = tf.cast(mask[:, :, i], 'float32')  # (None,1), alive=1, dead/dummy=0

            q1 = q1 * float_mask
            q2 = q2 * float_mask

            q1s.append(q1)  # [(None,action_dim),...], len=n
            q2s.append(q2)

        return q1s, q2s  # [(None,action_dim),...], len=n

    @tf.function
    def call(self, x, masks, training=False):
        """
        :x=[agents_obs, global_state]
            :agents_obs: [(None,g,g,ch*n_frames),...], len=n
            :global_obs: (None,g,g,global_ch*global_n_frames),
        :masks: [(None,1,n),...], len=n

        :return: policy_logits: [(None,action_dim),...], len=n
                 q1s, q2s: [(None,action_dim),...], len=n
                 att_scores=[scores1,scores2]: [[(None,num_heads,n),...],[(None,num_heads,n),...]]
        """
        agents_obs = x[0]  # [(None,15,15,6),...], len=n
        global_state = x[1]  # (None,15,15,6)

        """ Global feature by Global CNN layer """
        global_feature = self.global_cnn(global_state)  # (None,hidden_dim)

        """ CNN layer """
        features_cnn = \
            self.cnn_block(agents_obs, masks, training=training)  # [(None,hidden_dim),...], len=n

        """ Dropout layer """
        features_dropout = \
            self.dropout_block(features_cnn, training=training)  # [(None,hidden_dim),...], len=n

        """ Multi Head Self-Attention layer 1 """
        features_mha1, scores1 = \
            self.mha_block(self.mha1, features_dropout, masks, training=training)
        # [(None,hidden_dim),...], [(None,num_heads,n),...], len=n

        """ Multi Head Self-Attention layer 2 """
        features_mha2, scores2 = \
            self.mha_block(self.mha2, features_mha1, masks, training=training)
        # [(None,hidden_dim),...], [(None,num_heads,n),...], len=n

        att_scores = [scores1, scores2]  # [[(None,num_heads,n),...],[(None,num_heads,n),...]]

        """ Policy (policy_probs output) """
        policy_logits = \
            self.policy_block(features_mha2, masks, training=training)
        # [(None,action_dim),...], len=n

        """ [q1s, q2s] """
        q1s, q2s = self.value_block(features_mha2, global_feature, masks, training=training)
        # [(None,action_dim),...], [(None,action_dim),...], len=n

        return [policy_logits, [q1s, q2s]], att_scores

    def process_action(self, action_logits, masks):
        """
        :param  action_logits: [(None,action_dim),...], len=n
                masks: [(None,1,n),...], len=n
        :return:  [(None,action_dim),...], len=n

        ※ probs=0, log_probs=0 of dead / dummy agents
        """

        probs = []
        log_probs = []

        for i in range(self.config.max_num_red_agents):
            mask = masks[i]  # (None,1,n)
            float_mask = tf.cast(mask[:, :, i], 'float32')  # (None,1), alive=1, dead/dummy=0

            prob = tf.nn.softmax(action_logits[i])  # (None,action_dim)
            prob = prob * float_mask  # (None,action_dim)

            log_prob = tf.nn.log_softmax(action_logits[i])  # (None,action_dim)
            log_prob = log_prob * float_mask  # (None,action_dim)

            probs.append(prob)  # [(None,action_dim),...], len=n
            log_probs.append(log_prob)  # [(None,action_dim),...], len=n

        return probs, log_probs  # [(None,action_dim),...], [(None,action_dim),..., len=n

    def sample_actions(self, x, masks, training=False):
        """ Use only agents obs.
        x : agents_obs, # [(None,g,g,ch*n_frames),...], len=n
        masks: # [(None,1,n),...], len=n

        action=5 if policyprobs=[0,0,0,0,0], that is or the dead or dummy agents
        :return: actions: [(None,1),...], len=n
                 scores: [[(None,num_heads,n),...],[(None,num_heads,n),...]]
        """

        features_cnn = self.cnn_block(x, masks, training=training)  # [(None,hidden_dim),...]

        features_dropout = \
            self.dropout_block(features_cnn, training=training)  # [(None,hidden_dim),...]

        features_mha1, scores1 = \
            self.mha_block(
                self.mha1, features_dropout, masks, training=training)
        # [(None,hidden_dim),...], [(None,num_heads,n),...]

        features_mha2, scores2 = \
            self.mha_block(
                self.mha2, features_mha1, masks, training=training)
        # [(None,hidden_dim),...], [(None,num_heads,n),...]

        scores = [scores1, scores2]  # [[(None,hidden_dim),...], [(None,num_heads,n),...]]

        policy_logits = \
            self.policy_block(features_mha2, masks, training=training)  # [(None,action_dim),...]

        policy_probs, _ = self.process_action(policy_logits, masks)  # [(None,action_dim),...]

        num_agents = self.config.max_num_red_agents
        actions = []

        for i in range(num_agents):
            cdist = tfp.distributions.Categorical(probs=policy_probs[i])
            action = cdist.sample()  # (None,1)
            actions.append(np.expand_dims(action.numpy(), axis=-1))

        return actions, scores
        # actions: [(None,1),...], lne=n, action=5 for the dead or dummy agents, int32
        # scores=[score1, score2]: [[(None,num_heads,n),...], [(None,num_heads,n),...]]

    def build_graph(self, masks):
        x = [tf.keras.layers.Input(
            shape=(self.config.grid_size,
                   self.config.grid_size,
                   self.config.observation_channels * self.config.n_frames))
            for _ in range(self.config.max_num_red_agents)]

        global_x = tf.keras.layers.Input(
            shape=(self.config.grid_size,
                   self.config.grid_size,
                   self.config.global_observation_channels * self.config.global_n_frames)
        )

        global_feature = self.global_cnn(global_x)

        features_cnn = self.cnn_block(x, masks)

        features_dropout = self.dropout_block(features_cnn, training=False)

        features_mha1, score1 = self.mha_block(self.mha1, features_dropout, masks, training=False)
        features_mha2, score2 = self.mha_block(self.mha2, features_mha1, masks, training=False)

        policy_logits = self.policy_block(features_mha2, masks, training=False)
        q1s, q2s = self.value_block(features_mha2, global_feature, masks, training=False)

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
    agents_obs = []

    masks = make_mask(alive_agents_ids, max_num_agents)  # [(b,1,n),...],len=n, bool

    for i in range(max_num_agents):
        if i in alive_agents_ids:
            obs = np.random.random(obs_shape).astype(np.float32)  # (g,g,ch*n_frames)
        else:
            obs = np.zeros(obs_shape).astype(np.float32)  # (g,g,ch*n_frames)

        obs = np.expand_dims(obs, axis=0)  # add batch_dim, (1,g,g,ch*n_frames)
        agents_obs.append(obs)

    """ Make model """
    marl_transformer = MarlTransformerGlobalStateModel(config=config)

    [policy_logits, [q1s, q2s]], scores = \
        marl_transformer([agents_obs, global_state], masks, training=False)

    marl_transformer.summary()

    """ Summary """
    """
    print('\n')
    print('--------------------------------------- model ---------------------------------------')
    # marl_transformer.build_graph(masks).summary()

    tf.keras.utils.plot_model(
        marl_transformer.build_graph(masks),
        to_file=dir_name + '/mtd_with_global_state',
        show_shapes=True,
        show_layer_activations=True,
        show_dtype=True,
        dpi=96 * 3
    )
    """

    probs, log_probs = marl_transformer.process_action(policy_logits, masks)
    # probs, log_probs: [(None,action_dim),...], len=n

    print('\n')
    for i in range(config.max_num_red_agents):
        print(f"probs[{i}]: {probs[i]}")
        print(f"log_probs[{i}]: {log_probs[i]}")

    """ Sample actions """
    actions, scores = marl_transformer.sample_actions(agents_obs, masks, training=False)
    # actions: [(None,1),...], lne=n, action=5 for the dead or dummy agents, int32
    # scores=[score1, score2]: [[(None,num_heads,n),...], [(None,num_heads,n),...]]

    print('\n')
    for i in range(config.max_num_red_agents):
        print(f"probs[{i}]: {probs[i]}")
        print(f"actions[{i}]: {actions[i]}")


if __name__ == '__main__':
    main()
