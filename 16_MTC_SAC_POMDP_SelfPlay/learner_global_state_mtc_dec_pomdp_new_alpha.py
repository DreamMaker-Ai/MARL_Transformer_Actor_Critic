from pathlib import Path

import numpy as np
import ray
import tensorflow as tf

from battlefield_strategy_team_reward_global_state import BattleFieldStrategy
from config_dec_pomdp import Config  # for build
from global_models_dec_pomdp import GlobalCNNModel  # for build
from models_global_state_mtc_dec_pomdp import MarlTransformerGlobalStateModel
from utils_transformer_mtc_dec_pomdp import make_mask, make_po_attention_mask, \
    make_padded_obs, make_padded_pos


@ray.remote
# @ray.remote(num_cpus=1, num_gpus=1)
class Learner:
    def __init__(self):
        self.env = BattleFieldStrategy()

        self.action_space_dim = self.env.action_space.n
        self.gamma = self.env.config.gamma

        self.mtc = MarlTransformerGlobalStateModel(config=self.env.config)

        self.target_mtc = MarlTransformerGlobalStateModel(config=self.env.config)

        self.logalpha = tf.Variable(tf.math.log(0.5))

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
            make_padded_obs(max_num_agents, obs_shape,
                            agent_obs)  # (1,n,2*fov+1,2*fov+1,ch*n_frames)

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

        self.mtc([[padded_obs, padded_pos], global_state],
                 mask, attention_mask, training=False)
        self.target_mtc([[padded_obs, padded_pos], global_state],
                        mask, attention_mask, training=False)

        # Load weights & alpha
        if self.env.config.model_dir:
            self.mtc.load_weights(self.env.config.model_dir)

        # if self.env.config.alpha_dir:
        #     logalpha = np.load(self.env.config.alpha_dir)
        #     self.logalpha = tf.Variable(logalpha)

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
                        (padded_)obss,  # (1,n,2*fov+1,2*fov+1,ch*n_frames)
                        (padded_)poss  # (1,n,2*n_frames)
                        (padded_)actions,  # (1,n)
                        (padded_)rewards,  # (1,n)
                        next_(padded_)obss,  # (1,n,2*fov+1,2*fov+1,ch*n_frames)
                        next_(padded_)poss  # (1,n,2*n_frames) 
                        (padded_)dones,  # (1,n), bool
                        team_reward,  # (1,1)
                        team_done,  # (1,1), bool
                        masks,  # (1,n), bool
                        next_masks,  # (1,n), bool
                        attention_masks,  # (1,n,n), bool
                        next_attention_masks,  # (1,n,n), bool
                        global_state,  # (1,g,g,global_ch*global_n_frames)
                        next_global_state,  # (1,g,g,global_ch*global_n_frames)
                    )

                ※ experience.states等で読み出し

        :return:
            current_weights: 最新のnetwork weights
        """
        q_losses = []  # policy loss
        p_losses = []  # Q loss
        alpha_losses = []  # entropy alpha loss

        for minibatch in minibatchs:
            # minibatchをnetworkに入力するshapeに変換

            # process in minibatch
            obss = []
            poss = []
            actions = []
            rewards = []
            next_obss = []
            next_poss = []
            dones = []
            team_reward = []
            team_done = []
            masks = []
            next_masks = []
            attention_masks = []
            next_attention_masks = []
            global_state = []
            next_global_state = []

            for i in range(len(minibatch)):
                obss.append(minibatch[i].obss)
                poss.append(minibatch[i].poss)
                actions.append(minibatch[i].actions)
                rewards.append(minibatch[i].rewards)
                next_obss.append(minibatch[i].next_obss)
                next_poss.append(minibatch[i].next_poss)
                dones.append(minibatch[i].dones)
                team_reward.append(minibatch[i].global_r)
                team_done.append(minibatch[i].global_done)
                masks.append(minibatch[i].masks)
                next_masks.append(minibatch[i].next_masks)
                attention_masks.append(minibatch[i].attention_masks)
                next_attention_masks.append(minibatch[i].next_attention_masks)
                global_state.append(minibatch[i].global_state)
                next_global_state.append(minibatch[i].next_global_state)

            # list -> ndarray
            obss = np.vstack(obss)  # (b,n,2*fov+1,2*fov+1,ch*n_frames)
            poss = np.vstack(poss)  # (b,n,2*n_frames)
            actions = np.vstack(actions)  # (b,n)
            rewards = np.vstack(rewards)  # (b,n)
            next_obss = np.vstack(next_obss)  # (b,n,2*fov+1,2*fov+1,ch*n_frames)
            next_poss = np.vstack(next_poss)  # (b,n,2*n_frames)
            dones = np.vstack(dones)  # (b,n), bool
            team_reward = np.vstack(team_reward)  # (b,1)
            team_done = np.vstack(team_done)  # (b,1)
            masks = np.vstack(masks)  # (b,n), bool
            next_masks = np.vstack(next_masks)  # (b,n), bool
            attention_masks = np.vstack(attention_masks)  # (b,n,n), bool
            next_attention_masks = np.vstack(next_attention_masks)  # (b,n,n), bool
            global_state = np.vstack(global_state)  # (b,g,g,global_ch*global_n_frames)
            next_global_state = np.vstack(next_global_state)  # (b,g,g,global_ch*global_n_frames)

            # ndarray -> tf.Tensor
            obss = tf.convert_to_tensor(obss, dtype=tf.float32)  # (b,n,2*fov+1,2*fov+1,ch*n_frames)
            poss = tf.convert_to_tensor(poss, dtype=tf.float32)  # (b,n,2*n_frames)
            actions = tf.convert_to_tensor(actions, dtype=tf.int32)  # (b,n)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)  # (b,n)
            next_obss = tf.convert_to_tensor(next_obss, dtype=tf.float32)
            # (b,n,2*fov+1,2*fov+1,ch*n_frames)
            next_poss = tf.convert_to_tensor(next_poss, dtype=tf.float32)  # (b,n,2*n_frames)
            dones = tf.convert_to_tensor(dones, dtype=tf.float32)  # (b,n), bool->float32
            masks = tf.convert_to_tensor(masks, dtype=tf.float32)  # (b,n), bool->float32
            team_reward = tf.convert_to_tensor(team_reward, dtype=tf.float32)  # (b,1)
            team_done = tf.convert_to_tensor(team_done, dtype=tf.float32)  # (b,1), bool->float32
            next_masks = tf.convert_to_tensor(next_masks, dtype=tf.float32)  # (b,n), bool->float32
            attention_masks = tf.convert_to_tensor(attention_masks, dtype=tf.float32)
            # (b,n,n), bool->float32
            next_attention_masks = tf.convert_to_tensor(next_attention_masks, dtype=tf.float32)
            # (b,n,n), bool->float32
            global_state = tf.convert_to_tensor(global_state, dtype=tf.float32)
            # (b,g,g,global_ch*global_n_frames)
            next_global_state = tf.convert_to_tensor(next_global_state, dtype=tf.float32)
            # (b,g,g,global_ch*global_n_frames)

            num_alive_agents = tf.reduce_sum(masks, axis=-1)  # (b,)

            alpha = tf.math.exp(self.logalpha)

            """ Update MTC """
            # Target valueの計算
            [next_action_logits, [next_q1, next_q2]], _ = \
                self.target_mtc([[next_obss, next_poss], next_global_state],
                                next_masks, next_attention_masks, training=False)
            # (b,n,action_dim), [(b,n,action_dim). (b,n,action_dim)]

            next_action_probs, next_action_logprobs = \
                self.target_mtc.process_action(next_action_logits, next_masks)
            # (b,n,action_dim)

            next_q = tf.math.minimum(next_q1, next_q2)  # (b,n,action_dim)

            v = tf.einsum('ijk,ijk->ij',
                          next_action_probs, next_q - alpha * next_action_logprobs)  # (b,n)

            targets = rewards + (1. - dones) * self.gamma * v  # (b,n)

            actions_onehot = tf.one_hot(actions, self.action_space_dim)  # (b,n,action_dim)

            with tf.GradientTape() as tape:

                """ Critic Q loss """
                [action_logits, [q1, q2]], _ = \
                    self.mtc([[obss, poss], global_state], masks, attention_masks, training=False)
                # (b,n,action_dim) [(b,n,action_dim), (b,n,action_dim)]

                action_probs, action_logprobs = self.mtc.process_action(action_logits, masks)
                # (b,n,action_dim), (b,n,action_dim)

                vpred1 = tf.reduce_sum(actions_onehot * q1, axis=-1)  # (b,n)
                vpred2 = tf.reduce_sum(actions_onehot * q2, axis=-1)  # (b,n)

                td1 = (targets - vpred1) * masks  # (b,n), td of alive agents
                td2 = (targets - vpred2) * masks  # (b,n)

                critic1_loss = tf.reduce_sum(tf.square(td1), axis=-1)  # (b,)
                critic1_loss = critic1_loss / num_alive_agents  # (b,)

                critic2_loss = tf.reduce_sum(tf.square(td2), axis=-1)  # (b,)
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
        if self.count % 5000 == 0:  # Default=5000
            save_dir = Path(__file__).parent / 'models'

            save_name = '/model_' + str(self.count) + '/'
            self.mtc.save_weights(str(save_dir) + save_name)

            save_name = '/alpha_' + str(self.count)
            logalpha = self.logalpha.numpy()
            np.save(str(save_dir) + save_name, logalpha)

        self.count += 1

        return current_weights, np.mean(p_loss), np.mean(q_loss), np.mean(alpha_loss)
