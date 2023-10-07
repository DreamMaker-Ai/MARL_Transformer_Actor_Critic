import ray
import numpy as np
import tensorflow as tf

from collections import deque

from models import MarlTransformerModel
from battlefield_strategy import BattleFieldStrategy
from utils_gnn import get_alive_agents_ids
from utils_transformer import make_mask, make_padded_obs


# @ray.remote(num_cpus=1, num_gpus=0)  # cloud使用時
@ray.remote
class Worker:
    def __init__(self, worker_id):
        """
        batch_size=t_max
        """
        self.worker_id = worker_id

        self.env = BattleFieldStrategy()
        self.gamma = self.env.config.gamma
        self.value_loss_coef = self.env.config.value_loss_coef
        self.entropy_coef = self.env.config.entropy_coef
        self.loss_coef = self.env.config.loss_coef
        self.batch_size = self.env.config.batch_size
        self.n_frames = self.env.config.n_frames

        self.action_space_dim = self.env.action_space.n

        # Make a policy network
        self.policy = MarlTransformerModel(config=self.env.config)

        self.obs_shape = (self.env.config.grid_size,
                          self.env.config.grid_size,
                          self.env.config.observation_channels * self.n_frames)

        # Initialize environment
        ### The followings are reset in 'reset_states'
        self.frames = None  # For each agent in env
        self.states = None
        self.prev_actions = None

        self.alive_agents_ids = None  # For all agents, including dummy ones
        self.padded_states = None
        self.padded_prev_actions = None
        self.mask = None

        self.episode_return = None
        self.step = None

        ### Initialize above Nones
        observations = self.env.reset()
        self.reset_states(observations)

        self.policy(self.padded_states, self.mask, training=True)  # build

    def reset_states(self, observations):
        # TODO prev_actions
        """
        alive_agents_ids: list of alive agent id

        # For agents in Env
             each agent stacks observations n-frames in channel-dims
             -> observations[red.id]: (grid_size,grid_size,channels)

             -> generate deque of length=n_frames
             self.frames[red.id]: deque[(grid_size,grid_size,channels),...]

             -> transform to states
             states[red.id]: (grid_size,grid_size,channels*n_frames)

             self.prev_actions[red.id]: int (TODO)
        """

        self.frames = {}
        self.states = {}
        self.prev_actions = {}

        for red in self.env.reds:
            # all reds are alive when reset

            self.frames[red.id] = deque([observations[red.id]] * self.n_frames,
                                        maxlen=self.n_frames)
            # [(grid_size,grid_size,channels),...,(grid_size,grid_size,channels)]

            self.states[red.id] = np.concatenate(self.frames[red.id], axis=2).astype(np.float32)
            # (grid_size,grid_size,channels*n_frames)

            # self.prev_actions[red.id] = 0

        # alive_agents_ids: list of alive agent id, [int,...], len=num_alive_agents
        self.alive_agents_ids = get_alive_agents_ids(env=self.env)

        # Get padded observations ndarray for all agents, including dead and dummy agents
        self.padded_states = \
            make_padded_obs(max_num_agents=self.env.config.max_num_red_agents,
                            obs_shape=self.obs_shape,
                            raw_obs=self.states)  # (1,n,g,g,ch*n_frames)

        # Get mask for the padding
        self.mask = make_mask(alive_agents_ids=self.alive_agents_ids,
                              max_num_agents=self.env.config.max_num_red_agents)  # (1,n)

        # reset episode variables
        self.episode_return = 0
        self.step = 0

    def rollout_and_compute_grads(self, weights):
        """
        0. Global policyの重みをコピー
        1. Rolloutして、batch_size分のデータを収集
        2. 収集したバッチデータからロスを算出して勾配を計算
        """

        """ 
        0. Global policyの重みをコピー 
        """
        self.policy.set_weights(weights=weights)

        """ 
        1. Rolloutして、batch_size分のデータを収集
            batch_size = sequence_length = b
            max_num_red_agents = n
            trajectory["s"]: (b,n,g,g,ch*n_frames)
            trajectory["a"]: (b,n), np.int32
            trajectory["r"]: (b,n)
            trajectory["dones"]: (b,n), bool
            trajectory["s2"]: next_states, (b,n,g,g,ch*n_frames)
            trajectory["mask"]: (b,n), bool
            trajectory["mask2"]: next_mask, (b,n), bool
            trajectory["R"]: (b,n)
        """

        trajectory = self._rollout()

        """ 2. 収集したバッチデータからロスを算出して勾配を計算 """
        with tf.GradientTape() as tape:
            [policy_probs, values], _ = \
                self.policy(trajectory["s"], trajectory["mask"], training=False)
            # (b,n,action_dim), (b,n,1)

            """ Compute log π(a|s) """
            selected_actions = tf.convert_to_tensor(trajectory["a"], dtype=tf.int32)  # (b,n)
            # one_hot for dead or dummy agents' action (=-1) is zero vectors.
            selected_actions_onehot = \
                tf.one_hot(selected_actions, depth=self.action_space_dim, dtype=tf.float32)
            # (b,n,action_dim)

            log_probs = \
                selected_actions_onehot * tf.math.log(policy_probs + 1e-5)  # (b,n,action_dim)
            selected_actions_log_probs = tf.reduce_sum(log_probs, axis=-1)  # (b,n)

            """ Covert trajectory["mask"] to tf.tensor (float32) """
            masks = tf.convert_to_tensor(trajectory["mask"], dtype=tf.float32)  # (b,n)

            """ Compute advantage and value loss """
            # Compute num of alive agents every batch (time step)
            num_alive_agents = tf.reduce_sum(masks, axis=-1)  # (b,)

            advantages = trajectory["R"] - tf.squeeze(values, axis=-1)  # (b,n)
            advantages = masks * advantages  # (b,n)

            value_loss = tf.reduce_sum(advantages ** 2, axis=-1)  # (b,)
            value_loss = value_loss / num_alive_agents  # (b,)
            value_loss = tf.reduce_mean(value_loss)

            mean_advantage = tf.reduce_mean(advantages)  # 表示用

            """ Compute policy loss """
            policy_loss = selected_actions_log_probs * tf.stop_gradient(advantages)  # (b,n)
            policy_loss = masks * policy_loss  # (b,n)
            policy_loss = tf.reduce_sum(policy_loss, axis=-1)  # (b,)
            policy_loss = policy_loss / num_alive_agents  # (b,)
            policy_loss = tf.reduce_mean(policy_loss)

            """ Compute entropy """
            entropy = - policy_probs * tf.math.log(policy_probs + 1e-5)  # (b,n,action_dim)
            entropy = tf.reduce_mean(entropy, axis=-1)  # (b,n)
            entropy = masks * entropy  # (b,n)
            entropy = tf.reduce_sum(entropy, axis=-1)  # (b,)
            entropy = entropy / num_alive_agents  # (b,)
            entropy = tf.reduce_mean(entropy)

            """ Compute total loss """
            loss = self.value_loss_coef * value_loss - 1 * policy_loss - \
                   1 * self.entropy_coef * entropy

            loss = self.loss_coef * loss

        grads = tape.gradient(loss, self.policy.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 30)  # default=40->10

        info = {"id": self.worker_id,
                "policy_loss": -1 * policy_loss * self.loss_coef,
                "value_loss": self.value_loss_coef * value_loss * self.loss_coef,
                "entropy": -1 * self.entropy_coef * entropy * self.loss_coef,
                "advantage": mean_advantage}

        return grads, info

    def _rollout(self):
        """
        Rolloutにより、t_start<=t<t_max 間(batch_size間)の
        {s,a,s',r,done,mask,mask',R(n-step return)}を取得
        """
        # 1. 初期化
        trajectory = {}
        trajectory["s"] = []
        trajectory["a"] = []
        trajectory["r"] = []
        trajectory["s2"] = []
        trajectory["dones"] = []
        trajectory["mask"] = []
        trajectory["mask2"] = []  # next_mask
        trajectory["R"] = []

        dones = {'all_dones': False}
        i = 0

        # 2. Rollout実施
        while not dones["all_dones"] and i < self.batch_size:
            # acts: action=-1 for the dead or dummy agents.
            acts = self.policy.sample_actions(
                self.padded_states, self.mask, training=False)  # (1,n), int32

            # get alive_agents & all agents actions.
            # * padded_actions: action=-1 for dead or dummy agents for
            #       utilizing tf.one_hot(-1) is zero vector
            # * actions['red_a'], a=alive agent id
            actions = {}  # For alive agents
            padded_actions = - np.ones((1, self.env.config.max_num_red_agents))  # (1,n)

            for idx in self.alive_agents_ids:
                agent_id = 'red_' + str(idx)
                actions[agent_id] = acts[0, idx]

                padded_actions[0, idx] = actions[agent_id]

            # One step of Lanchester simulation, for alive agents in env
            next_obserations, rewards, dones, infos = self.env.step(actions)

            # Make next_agents_states, next_agents_adjs, and next_alive_agents_ids,
            # including dummy ones
            next_alive_agents_ids = get_alive_agents_ids(env=self.env)

            ### For alive agents in env
            next_states = {}

            for idx in next_alive_agents_ids:
                agent_id = 'red_' + str(idx)

                self.frames[agent_id].append(
                    next_obserations[agent_id]
                )  # append (g,g,ch) to deque

                next_states[agent_id] = np.concatenate(
                    self.frames[agent_id], axis=2
                ).astype(np.float32)  # (g,g,ch*n_frames)

            # Get padded next observations ndarray of all agent
            next_padded_states = \
                make_padded_obs(
                    max_num_agents=self.env.config.max_num_red_agents,
                    obs_shape=self.obs_shape,
                    raw_obs=next_states
                )  # (1,n,g,g,ch*n_frames)

            # Get next mask for the padding
            next_mask = \
                make_mask(
                    alive_agents_ids=next_alive_agents_ids,
                    max_num_agents=self.env.config.max_num_red_agents
                )  # (1,n)

            # 終了判定
            if self.step > self.env.config.max_steps:

                for idx in self.alive_agents_ids:
                    agent_id = 'red_' + str(idx)
                    dones[agent_id] = True

                dones['all_dones'] = True

            # agents_rewards and agents_dones, including dead and dummy ones
            # reward = 0 and done=True for dead or dummy agents.
            agents_rewards = []
            agents_dones = []

            for idx in range(self.env.config.max_num_red_agents):
                if idx in self.alive_agents_ids:
                    agent_id = 'red_' + str(idx)
                    agents_rewards.append(float(rewards[agent_id]))
                    agents_dones.append(dones[agent_id])
                else:
                    agents_rewards.append(0.0)
                    agents_dones.append(True)

            # if len(agents_rewards) != self.env.config.max_num_red_agents:
            #     raise ValueError()

            # if len(agents_dones) != self.env.config.max_num_red_agents:
            #     raise ValueError()

            # Update episode return
            self.episode_return += np.sum(agents_rewards)

            # list -> ndarray
            padded_rewards = np.stack(agents_rewards, axis=0)  # (n,)
            padded_rewards = np.expand_dims(padded_rewards, axis=0)  # (1,n)

            padded_dones = np.stack(agents_dones, axis=0)  # (n,), bool
            padded_dones = np.expand_dims(padded_dones, axis=0)  # (1,n)

            # alive_agents_ids = np.array(self.alive_agents_ids, dtype=object)  # (a,), object
            # alive_agents_ids = np.expand_dims(alive_agents_ids, axis=0)  # (1,a)

            trajectory["s"].append(self.padded_states)  # append (1,n,g,g,ch*n_frames)
            trajectory["a"].append(padded_actions)  # append (1,n)
            trajectory["r"].append(padded_rewards)  # append (1,n)
            trajectory["s2"].append(next_padded_states)  # append (1,n,g,g,ch*n_frames)
            trajectory["dones"].append(padded_dones)  # append (1,n)
            trajectory["mask"].append(self.mask)  # append (1,n)
            trajectory["mask2"].append(next_mask)  # append (1,n)

            if dones['all_dones']:
                # print(f'episode reward = {self.episode_return}')
                observations = self.env.reset()
                self.reset_states(observations)
            else:
                self.alive_agents_ids = next_alive_agents_ids
                self.padded_states = next_padded_states
                self.mask = next_mask

                self.step += 1

            i += 1

        trajectory["s"] = np.concatenate(trajectory["s"], axis=0).astype(np.float32)
        # (b,n,g,g,ch*n_frames)
        trajectory["a"] = np.concatenate(trajectory["a"], axis=0).astype(np.int32)
        # (b,n), np.int32
        trajectory["r"] = np.concatenate(trajectory["r"], axis=0).astype(np.float32)
        # (b,n)
        trajectory["s2"] = np.concatenate(trajectory["s2"], axis=0).astype(np.float32)
        # (b,n,g,g,ch*n_frames)
        trajectory["dones"] = np.concatenate(trajectory["dones"], axis=0).astype(bool)
        # (b,n), bool
        trajectory["mask"] = np.concatenate(trajectory["mask"], axis=0).astype(bool)
        # (b,n), bool
        trajectory["mask2"] = np.concatenate(trajectory["mask2"], axis=0).astype(bool)
        # (b,n), bool

        # 3. Multi-step discounted returnをバッチ的に計算
        # values = (v(s_1), ..., v(s_tmax)), (b,n,1)
        # v(s_tmax) = vales[-1], (n,1)
        trajectory["R"] = []
        [_, values], _ = self.policy(trajectory["s2"], trajectory["mask2"], training=False)

        R = values[-1, :, 0]  # (n,)

        seq_len = trajectory["s"].shape[0]

        for i in reversed(range(seq_len)):
            R = trajectory["r"][i] + \
                self.gamma * (1. - trajectory["dones"][i].astype(np.float32)) * R  # (n,)
            trajectory["R"].insert(0, np.expand_dims(R, axis=0))  # insert (1,n) to the head

        trajectory["R"] = np.concatenate(trajectory["R"], axis=0).astype(np.float32)  # (b,n)

        return trajectory
