from pathlib import Path

import numpy as np
import ray
import tensorflow as tf

from battlefield_strategy import BattleFieldStrategy
from models import MarlTransformerModel
from utils_transformer import make_mask, make_padded_obs


@ray.remote
# @ray.remote(num_cpus=1, num_gpus=1)
class Learner:
    def __init__(self):
        self.env = BattleFieldStrategy()

        self.action_space_dim = self.env.action_space.n
        self.gamma = self.env.config.gamma

        self.mtc = MarlTransformerModel(config=self.env.config)

        self.target_mtc = MarlTransformerModel(config=self.env.config)

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

        # Build graph with dummy inputs
        grid_size = self.env.config.grid_size
        ch = self.env.config.observation_channels
        n_frames = self.env.config.n_frames

        obs_shape = (grid_size, grid_size, ch * n_frames)

        max_num_agents = self.env.config.max_num_red_agents

        alive_agents_ids = [0, 2]

        raw_obs = {}

        for i in alive_agents_ids:
            agent_id = 'red_' + str(i)
            raw_obs[agent_id] = np.random.rand(obs_shape[0], obs_shape[1], obs_shape[2])

        # Get padded_obs and mask
        padded_obs = make_padded_obs(max_num_agents=max_num_agents,
                                     obs_shape=obs_shape,
                                     raw_obs=raw_obs)  # (1,n,g,g,ch*n_frames)

        mask = make_mask(alive_agents_ids=alive_agents_ids,
                         max_num_agents=max_num_agents)  # (1,n)

        self.mtc(padded_obs, mask, training=False)
        self.target_mtc(padded_obs, mask, training=False)

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
                        (padded_)states,  # (1,n,g,g,ch*n_frames)
                        (padded_)actions,  # (1,n)
                        (padded_)rewards,  # (1,n)
                        next_(padded_)states,  # (1,n,g,g,ch*n_frames)
                        (padded_)dones,  # (1,n), bool
                        masks,  # (1,n), bool
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
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []
            masks = []
            next_masks = []

            for i in range(len(minibatch)):
                states.append(minibatch[i].states)
                actions.append(minibatch[i].actions)
                rewards.append(minibatch[i].rewards)
                next_states.append(minibatch[i].next_states)
                dones.append(minibatch[i].dones)
                masks.append(minibatch[i].masks)
                next_masks.append(minibatch[i].next_masks)

            # list -> ndarray
            states = np.vstack(states)  # (b,n,g,g,ch*n_frames)
            actions = np.vstack(actions)  # (b,n)
            rewards = np.vstack(rewards)  # (b,n)
            next_states = np.vstack(next_states)  # (b,n,g,g,ch*n_frames)
            dones = np.vstack(dones)  # (b,n), bool
            masks = np.vstack(masks)  # (b,n), bool
            next_masks = np.vstack(next_masks)  # (b,n), bool

            # ndarray -> tf.Tensor
            states = tf.convert_to_tensor(states, dtype=tf.float32)  # (b,n,g,g,ch*n_frames)
            actions = tf.convert_to_tensor(actions, dtype=tf.int32)  # (b,n)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)  # (b,n)
            next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
            # (b,n,g,g,ch*n_frames)
            dones = tf.convert_to_tensor(dones, dtype=tf.float32)  # (b,n), bool->float32
            masks = tf.convert_to_tensor(masks, dtype=tf.float32)  # (b,n), bool->float32
            next_masks = tf.convert_to_tensor(next_masks, dtype=tf.float32)  # (b,n), bool->float32

            num_alive_agents = tf.reduce_sum(masks, axis=-1)  # (b,)

            alpha = tf.math.exp(self.logalpha)

            """ Update MTC """
            # Target valueの計算
            [next_action_logits, [next_q1, next_q2]], _ = \
                self.target_mtc(next_states, next_masks, training=False)
            # (b,n,action_dim), [(b,n,action_dim). (b,n,action_dim)]

            next_action_probs, next_action_logprobs = \
                self.target_mtc.process_action(next_action_logits, next_masks)
            # (b,n,action_dim)

            next_q = tf.math.minimum(next_q1, next_q2)  # (b,n,action_dim)

            targets = rewards + (1. - dones) * self.gamma * (
                tf.einsum('ijk,ijk->ij', next_action_probs, next_q - alpha * next_action_logprobs)
            )  # (b,n)

            actions_onehot = tf.one_hot(actions, self.action_space_dim)  # (b,n,action_dim)

            with tf.GradientTape() as tape:

                """ Critic Q loss """
                [action_logits, [q1, q2]], _ = self.mtc(states, masks, training=False)
                # (b,n,action_dim) [(b,n,action_dim), (b,n,action_dim)]

                action_probs, action_logprobs = self.mtc.process_action(action_logits, masks)
                # (b,n,action_dim), (b,n,action_dim)

                vpred1 = tf.reduce_sum(actions_onehot * q1, axis=-1)  # (b,n)
                vpred2 = tf.reduce_sum(actions_onehot * q2, axis=-1)  # (b,n)

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
        if self.count % 1000 == 0:  # Default=500
            save_dir = Path(__file__).parent / 'models'

            save_name = '/model_' + str(self.count) + '/'
            self.mtc.save_weights(str(save_dir) + save_name)

            save_name = '/alpha_' + str(self.count)
            logalpha = self.logalpha.numpy()
            np.save(str(save_dir) + save_name, logalpha)

        self.count += 1

        return current_weights, np.mean(p_loss), np.mean(q_loss), np.mean(alpha_loss)
