from pathlib import Path

import numpy as np
import ray
import tensorflow as tf
from collections import deque

from battlefield_strategy_team_reward_global_state import BattleFieldStrategy
from models_global_state_dec_pomdp import MarlTransformerGlobalStateModel
from utils_transformer_dec_pomdp import make_po_id_mask as make_mask
from utils_transformer_dec_pomdp import experiences2per_agent_list_dec_pomdp, \
    per_agent_list2input_list_dec_pomdp, get_td_mask
from utils_gnn import get_alive_agents_ids

@ray.remote
# @ray.remote(num_cpus=1, num_gpus=1)
class Learner:
    def __init__(self):
        self.env = BattleFieldStrategy()

        self.action_space_dim = self.env.action_space.n
        self.gamma = self.env.config.gamma

        self.mtc = MarlTransformerGlobalStateModel(config=self.env.config)

        self.target_mtc = MarlTransformerGlobalStateModel(config=self.env.config)

        self.logalpha = tf.Variable(0.0)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.env.config.learning_rate)

        self.alpha_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.env.config.alpha_learning_rate,
            clipvalue=self.env.config.alpha_clip,
        )

        self.target_entropy = -0.98 * np.log(self.action_space_dim)

        self.count = self.env.config.n0 + 1

    def define_network(self):
        """
        Q-network, Target_networkを定義し、current weightsを返す
        """
        # self.mtc.compile(optimizer=self.optimizer, loss='mse')

        dummy_env = BattleFieldStrategy()
        dummy_env.reset()
        config = dummy_env.config

        epsilon = 0.

        # Make dummy_policy and load learned weights
        dummy_policy = MarlTransformerGlobalStateModel(config=config)

        # Build model
        grid_size = config.grid_size

        global_ch = config.global_observation_channels
        global_n_frames = config.global_n_frames

        ch = config.observation_channels
        n_frames = config.n_frames
        max_num_agents = 15

        fov = config.fov
        com = config.com

        global_state_shape = (grid_size, grid_size, global_ch * global_n_frames)
        global_state = \
            np.random.rand(global_state_shape[0], global_state_shape[1], global_state_shape[2])

        # The tester does not use global state, but it is required for network instantiation.
        global_frames = deque([global_state] * global_n_frames, maxlen=global_n_frames)
        global_frames = np.concatenate(global_frames, axis=2).astype(np.float32)
        # (g,g,global_ch*global_n_frames)
        global_state = np.expand_dims(global_frames, axis=0)
        # (1,g,g,global_ch*global_n_frames)

        obs_shape = (2 * fov + 1, 2 * fov + 1, ch * n_frames)

        # Define alive_agents_ids & raw_obs
        alive_agents_ids = get_alive_agents_ids(env=dummy_env)
        agents_obs = []

        masks = make_mask(alive_agents_ids, max_num_agents, dummy_env.reds, com)
        # [(1,1,n),...],len=n, bool

        for i in range(max_num_agents):
            if i in alive_agents_ids:
                obs = np.random.random(obs_shape).astype(
                    np.float32)  # (2*fov+1,2*fov+1,ch*n_frames)
                pos = np.random.random((2 * n_frames,)).astype(np.float32)  # (2*n_frames,)
            else:
                obs = np.zeros(obs_shape).astype(np.float32)  # (2*fov+1,2*fov+1,ch*n_frames)
                pos = np.zeros((2 * n_frames,)).astype(np.float32)  # (2*n_frames,)

            obs = np.expand_dims(obs, axis=0)  # add batch_dim, (1,2*fov+1,2*fov+1,ch*n_frames)
            pos = np.expand_dims(pos, axis=0)  # (1,2*n_frames)
            agents_obs.append([obs, pos])

        self.mtc([agents_obs, global_state], masks, training=False)
        self.target_mtc([agents_obs, global_state], masks, training=False)

        # Load weights & alpha
        if self.env.config.model_dir:
            self.mtc.load_weights(self.env.config.model_dir)

        if self.env.config.alpha_dir:
            logalpha = np.load(self.env.config.alpha_dir)
            self.logalpha = tf.Variable(logalpha)

        # Q networkのCurrent weightsをget
        current_weights = self.mtc.get_weights()

        # Q networkの重みをTarget networkにコピー
        self.target_mtc.set_weights(current_weights)

        return [current_weights, self.logalpha]

    def update_network(self, minibatchs):
        """
        minicatchsを使ってnetworkを更新
        minibatchs = [minibatch,...], len=num_minibatchs(=3)
            minibatch = [experience,...], len=batch_size(=16)
                experience =
                    (
                    self.padded_obss,       # [(1,2*fov+1,2*fov+1,ch*n_frames),(1,2*n_frames)],...], 
                                                len=n
                    padded_actions,         # [(1,),...]
                    padded_rewards,         # [(1,),...]
                    next_padded_obss,       # [(1,2*fov+1,2*fov+1,ch*n_frames),(1,2*n_frames)],...]
                    padded_dones,           # [(1,),...], bool
                    global_r,               # team_r; (1,1)
                    global_done,            # team_done; (1,1), bool
                    self.masks,             # [(1,1,n),...], bool
                    next_masks,             # [(1,1,n),...], bool
                    self.global_state,      # (1,g,g,global_ch*global_n_frames)
                    next_global_state,      # (1,g,g,global_ch*global_n_frames)
                    )

                ※ experience.obss等で読み出し

        :return:
            current_weights: 最新のnetwork weights
        """
        q_losses = []  # Q loss
        p_losses = []  # policy loss
        alpha_losses = []  # entropy alpha loss

        for minibatch in minibatchs:
            # minibatchをnetworkに入力するshapeに変換

            """
            agent_obs = [
                obs list of agent_1: [(1,2*fov+1,2*fov+1,ch*n_frames),...], len=b
                obs list of agent_2: [(1,2*fov+1,2*fov+1,ch*n_frames),...], len=b
                ...
            ], len=n
            
            agent_pos = [
                pos list of agent_1: [(1,2*n_frames),...],len=b
                pos list of agent_2: [(1,2*n_frames),...],len=b
                ...
            ], len=n
            
            agent_action = [
                action list of agent_1: [(1,),...], len=b
                action list of agent_2: [(1,),...], len=b
                ...
            ], len=n
            
            agent_mask = [
                mask list of agent_1: [(1,1,n),...], len=b
                mask list of agent_2: [(1,1,n),...], len=b
                ...
            ], len=n, bool
            
            team_reward: [(1,1),...], len=b
            team_done: [(1,1),...], len=b, bool
            
            global_state: [(1,g,g,global_ch*global_n_frames),...], len=b
            """

            agent_obs, agent_pos, agent_action, agent_reward, agent_next_obs, agent_next_pos, \
            agent_done, team_r, team_done, agent_mask, agent_next_mask, global_state, \
            next_global_state = \
                experiences2per_agent_list_dec_pomdp(self, minibatch)

            """
            obss: [[(b,2*fov+1,2*fov+1,ch*n_frames),(b,2*n_frame)],...], len=n
            actions: [(b,),...], len=n, int32
            rewards: [(b,),...], len=n
            next_obss: [[(b,2*fov+1,2*fov+1,ch*n_frames),(b,2*n_frame)],...], len=n
            dones: [(b,),...], len=n, float32
            team_rs: (b,1)
            team_dones: (b,1), float32
            masks: [(b,1,n),...], len=n, float32
            next_masks: [(b,1,n),...], len=n, float32
            global_states: (b,g,g,global_ch*global_n_frames)
            next_global_states: (b,g,g,global_ch*global_n_frames)
            """

            obss, actions, rewards, next_obss, dones, team_rs, team_dones, masks, next_masks, \
            global_states, next_global_states = \
                per_agent_list2input_list_dec_pomdp(self, agent_obs, agent_pos, agent_action,
                                                    agent_reward, agent_next_obs, agent_next_pos,
                                                    agent_done, team_r, team_done, agent_mask,
                                                    agent_next_mask, global_state,
                                                    next_global_state)

            batch_size = self.env.config.batch_size

            num_alive_agents = np.zeros((batch_size, 1), dtype=np.float32)  # (b,1)
            for idx in range(self.env.config.max_num_red_agents):
                num_alive_agents += masks[idx][:, :, idx]  # (b,1)

            num_alive_agents = tf.squeeze(num_alive_agents, axis=-1)  # (b,)

            alpha = tf.math.exp(self.logalpha)

            """ Update MTC """
            # Target valueの計算
            [next_action_logits, [next_q1, next_q2]], _ = \
                self.target_mtc([next_obss, next_global_states], next_masks, training=False)
            # next_action_logits: [(b,action_dim),...], len=n
            # next_q1: [(b,action_dim),...], len=n
            # next_q2: [(b,action_dim),...], len=n

            next_action_probs, next_action_logprobs = \
                self.target_mtc.process_action(next_action_logits, next_masks)
            # next_action_probs, next_action_logprobs: [(b,action_dim),...], len=n

            # len=n list -> tf.tensor
            actions = tf.stack(actions, axis=1)  # (b,n)

            next_q1 = tf.stack(next_q1, axis=1)  # (b,n,action_dim)
            next_q2 = tf.stack(next_q2, axis=1)  # (b,n,action_dim)
            next_action_probs = tf.stack(next_action_probs, axis=1)  # (b,n,action_dim)
            next_action_logprobs = tf.stack(next_action_logprobs, axis=1)  # (b,n,action_dim)

            # compute next_masks
            next_masks = get_td_mask(self.env.config, next_masks)  # (b,n)

            next_q = tf.math.minimum(next_q1, next_q2)  # (b,n,action_dim)

            v = tf.einsum('ijk,ijk->ij',
                          next_action_probs, next_q - alpha * next_action_logprobs)  # (b,n)

            v = tf.reduce_sum(v * next_masks, axis=-1, keepdims=True)  # (b,1)

            targets = team_rs + (1. - team_dones) * self.gamma * v  # (b,1)

            actions_onehot = tf.one_hot(actions, self.action_space_dim)  # (b,n,action_dim)

            with tf.GradientTape() as tape:
                """ Critic Q loss """
                [action_logits, [q1, q2]], _ = \
                    self.mtc([obss, global_states], masks, training=False)
                # action_logits: [(b,action_dim),...], len=n
                # q1: [(b,action_dim),...], len=n
                # q2: [(b,action_dim),...], len=n

                action_probs, action_logprobs = self.mtc.process_action(action_logits, masks)
                # action_probs, action_logprobs: [(b,action_dim),...], len=n

                # list -> tf.tensor
                q1 = tf.stack(q1, axis=1)  # (b,n,action_dim)
                q2 = tf.stack(q2, axis=1)  # (b,n,action_dim)
                action_probs = tf.stack(action_probs, axis=1)  # (b,n,action_dim)
                action_logprobs = tf.stack(action_logprobs, axis=1)  # (b,n,action_dim)

                # compute masks
                masks = get_td_mask(self.env.config, masks)  # (b,n)

                vpred1 = tf.reduce_sum(actions_onehot * q1, axis=-1)  # (b,n)
                vpred2 = tf.reduce_sum(actions_onehot * q2, axis=-1)  # (b,n)

                vpred1 = tf.reduce_sum(vpred1 * masks, axis=-1, keepdims=True)  # (b,1)
                vpred2 = tf.reduce_sum(vpred2 * masks, axis=-1, keepdims=True)  # (b,1)

                critic1_loss = tf.reduce_sum(tf.square(targets - vpred1), axis=-1)  # (b,)
                critic1_loss = critic1_loss / num_alive_agents  # (b,)

                critic2_loss = tf.reduce_sum(tf.square(targets - vpred2), axis=-1)  # (b,)
                critic2_loss = critic2_loss / num_alive_agents  # (b,)

                q_loss1 = tf.reduce_mean(critic1_loss)
                q_loss2 = tf.reduce_mean(critic2_loss)

                q_loss = 0.5 * (q_loss1 + q_loss2)

                """ Policy loss """
                q = tf.stop_gradient(tf.math.minimum(q1, q2))  # (b,n,action_dim)

                policy_loss = tf.einsum('ijk,ijk->ij', action_probs, alpha * action_logprobs - q)
                # (b,n)
                policy_loss = tf.reduce_sum(policy_loss, axis=-1)  # (b,)
                p_loss = policy_loss / num_alive_agents  # (b,)

                p_loss = tf.reduce_mean(p_loss)

                """ Total loss of MTC """
                loss = q_loss + self.env.config.ploss_coef * p_loss

            variables = self.mtc.trainable_variables
            grads = tape.gradient(loss, variables)
            grads, _ = tf.clip_by_global_norm(grads, self.env.config.gradient_clip)

            self.optimizer.apply_gradients(zip(grads, variables))

            """ Update alpha """
            entropy_diff = - action_logprobs - self.target_entropy  # (b,n,action_dim)

            with tf.GradientTape() as tape:
                alpha_loss = tf.einsum('ijk,ijk->ij',
                                       action_probs,
                                       tf.math.exp(self.logalpha) * entropy_diff)  # (b,n)

                alpha_loss = tf.reduce_sum(alpha_loss, axis=-1)  # (b,)
                alpha_loss = alpha_loss / num_alive_agents  # (b,)

                alpha_loss = self.env.config.aloss_coef * tf.reduce_mean(alpha_loss)

            grads = tape.gradient(alpha_loss, self.logalpha)
            self.alpha_optimizer.apply_gradients([(grads, self.logalpha)])

            q_losses.append(q_loss)
            p_losses.append(p_loss)
            alpha_losses.append(alpha_loss)

        # 最新のネットワークweightsをget
        mtc_weights = self.mtc.get_weights()

        current_weights = [mtc_weights, self.logalpha]

        # Target networkのweights更新: Soft update
        target_weights = self.target_mtc.get_weights()

        for w in range(len(target_weights)):
            target_weights[w] = \
                self.env.config.tau * mtc_weights[w] + \
                (1. - self.env.config.tau) * target_weights[w]

        self.target_mtc.set_weights(target_weights)

        # Save model
        if self.count % 1000 == 0:  # Default=5000
            save_dir = Path(__file__).parent / 'models'

            save_name = '/model_' + str(self.count) + '/'
            self.mtc.save_weights(str(save_dir) + save_name)

            save_name = '/alpha_' + str(self.count)
            logalpha = self.logalpha.numpy()
            np.save(str(save_dir) + save_name, logalpha)

        self.count += 1

        return current_weights, np.mean(p_loss), np.mean(q_loss), np.mean(alpha_loss)
