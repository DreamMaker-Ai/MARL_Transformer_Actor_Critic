import ray
import numpy as np

from collections import deque

from models_global_state_mtc_dec_pomdp import MarlTransformerGlobalStateModel
from battlefield_strategy_pomdp_sp2 import BattleFieldStrategy
from utils_gnn import get_alive_agents_ids
from utils_transformer_mtc_dec_pomdp import make_mask, make_po_attention_mask, \
    make_padded_obs, make_padded_pos


# @ray.remote(num_cpus=1, num_gpus=0)  # cloud使用時
@ray.remote
class Worker:
    def __init__(self, pid):
        """
        batch_size=t_max
        """
        self.pid = pid

        self.env = BattleFieldStrategy()
        self.batch_size = self.env.config.batch_size
        self.n_frames = self.env.config.n_frames

        self.action_space_dim = self.env.action_space.n

        # Make a policy network
        self.mtc = MarlTransformerGlobalStateModel(config=self.env.config)

        self.global_n_frames = self.env.config.global_n_frames
        self.global_obs_shape = (self.env.config.grid_size,
                                 self.env.config.grid_size,
                                 self.env.config.global_observation_channels * self.global_n_frames)

        self.obs_shape = (2 * self.env.config.fov + 1,
                          2 * self.env.config.fov + 1,
                          self.env.config.observation_channels * self.n_frames)  # (5,5,16)

        self.pos_shape = (2 * self.env.config.n_frames,)  # (8,)

        # Define local buffer
        self.buffer = []

        # Initialize environment
        ### The followings are reset in 'reset_states'
        self.frames = None  # For each agent in env
        self.pos_frames = None
        self.prev_actions = None

        self.alive_agents_ids = None  # For all agents, including dummy ones
        self.padded_obss = None
        self.padded_poss = None
        self.padded_prev_actions = None
        self.mask = None
        self.attention_mask = None

        self.global_frames = None
        self.global_state = None

        self.episode_return = None
        self.step = None

        ### Initialize above Nones
        observations, global_observation = self.env.reset()
        self.reset_states(observations, global_observation)

        self.mtc([[self.padded_obss, self.padded_poss], self.global_state],
                 self.mask, self.attention_mask, training=True)  # build

    def reset_states(self, observations, global_observation):
        # TODO prev_actions
        """
        alive_agents_ids: list of alive agent id

        # For agents in Env
             each agent stacks observations and positions n-frames in channel-dims
             -> observations[red.id]: (2*fov+1,2*fov+1,channels)
                positions[red.id]: (2,)

             -> generate deque
             self.frames[red.id]: deque[(2*fov+1,2*fov+1,channels),...], len=n_frames
             self.pos_frames[red.id]: deque[(2,),...], len=n_frames

             -> transform to tensor
             obss[red.id]: (2*fov+1,2*fov+1,channels*n_frames)=(5,5,4*4)
             poss[red.id]: (2*n_frames,)=(2*4,)

             self.prev_actions[red.id]: int (TODO)
        """

        self.frames = {}
        self.pos_frames = {}
        obss = {}
        poss = {}
        self.prev_actions = {}

        for red in self.env.reds:
            # all reds are alive when reset

            self.frames[red.id] = deque([observations[red.id]] * self.n_frames,
                                        maxlen=self.n_frames)
            # [(2*fov+1,2*fov+1,channels),...], len=n_frames

            obss[red.id] = np.concatenate(self.frames[red.id], axis=2).astype(np.float32)
            # (2*fov+1,2*fov+1,channels*n_frames)=(5,5,16)

            self.pos_frames[red.id] = \
                deque([(red.pos[0] / self.env.config.grid_size,
                        red.pos[1] / self.env.config.grid_size)] * self.n_frames,
                      maxlen=self.n_frames)
            # [(2,),..., len=n_frames

            poss[red.id] = np.concatenate(self.pos_frames[red.id], axis=0).astype(np.float32)
            # (2*n_frames,)=(8,)

            # self.prev_actions[red.id] = 0

        # alive_agents_ids: list of alive agent id, [int,...], len=num_alive_agents
        self.alive_agents_ids = get_alive_agents_ids(env=self.env)

        # global state
        self.global_frames = deque(
            [global_observation] * self.global_n_frames,
            maxlen=self.global_n_frames
        )

        global_frames = np.concatenate(self.global_frames, axis=2).astype(np.float32)

        self.global_state = \
            np.expand_dims(global_frames, axis=0)  # (1,g,g,global_ch*global_n_frames)

        # Get padded observations ndarray for all agents, including dead and dummy agents
        self.padded_obss = \
            make_padded_obs(max_num_agents=self.env.config.max_num_red_agents,
                            obs_shape=self.obs_shape,
                            raw_obs=obss)  # (1,n,2*fov+1,2*fov+1,ch*n_frames)=(1,15,5,5,16)

        self.padded_poss = \
            make_padded_pos(max_num_agents=self.env.config.max_num_red_agents,
                            pos_shape=self.pos_shape,
                            raw_pos=poss)  # (1,n,2*n_frames)=(1,15,8)

        # Get mask for the padding
        self.mask = make_mask(alive_agents_ids=self.alive_agents_ids,
                              max_num_agents=self.env.config.max_num_red_agents)  # (1,n)

        # Get attention mask
        self.attention_mask = \
            make_po_attention_mask(
                alive_agents_ids=self.alive_agents_ids,
                max_num_agents=self.env.config.max_num_red_agents,
                agents=self.env.reds,
                com=self.env.config.com
            )  # (1,n,n)

        # reset episode variables
        self.episode_return = 0
        self.step = 0

    def rollout_and_collect_trajectory(self, weights):
        """
        0. Global policyの重みをコピー
        1. Rolloutして、batch_size分のデータを収集
        """

        """ 
        0. Global MTC  の重みをコピー 
        """
        self.mtc.set_weights(weights=weights[0])

        """ 
        1. Rolloutして、batch_size分のデータを収集
            batch_size = sequence_length = b
            max_num_red_agents = n
            transitions = [transition, ...], list
            transition = (
                self.padded_obss,  # append (1,n,2*fov+1,2*fov+1,ch*n_frames)
                self.padded_poss,  # append (1,n,2*n_frames)
                padded_actions,  # append (1,n)
                padded_rewards,  # append (1,n)
                next_padded_obss,  # append (1,n,2*fov+1,2*fov+1,ch*n_frames)
                next_padded_poss,  # append (1,n,2*n_frames)
                padded_dones,  # append (1,n), bool
                global_r,  # append (1,1)
                global_done,  # append (1,1), bool
                self.mask,  # append (1,n), bool
                next_mask,  # append (1,n), bool
                self.attention_mask,  # append (1,n,n), bool
                next_attention_mask,  # append (1,n,n), bool
                global_state,  # append (1,g,g,global_ch*global_n_frames)
                next_global_state,  # append (1,g,g,global_ch*global_n_frames)
            )
        """

        trajectory = self._rollout()

        return trajectory

    def _rollout(self):
        """
        Rolloutにより、t_start<=t<t_max 間(batch_size間)の
        {s,a,s',r,done,mask}を取得
        """

        # Rollout実施
        for i in range(self.batch_size):
            # acts: action=-1 for the dead or dummy agents.
            acts, _ = self.mtc.sample_actions(
                [self.padded_obss, self.padded_poss],
                self.mask, self.attention_mask, training=False)  # (1,n), int32

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
            next_obserations, rewards, dones, infos, reward, done, next_global_observation = \
                self.env.step(actions)

            # Make next_agents_states, next_agents_adjs, and next_alive_agents_ids,
            # including dummy ones
            next_alive_agents_ids = get_alive_agents_ids(env=self.env)

            ### For alive agents in env
            next_obss = {}
            next_poss = {}

            for idx in next_alive_agents_ids:
                agent_id = 'red_' + str(idx)

                self.frames[agent_id].append(
                    next_obserations[agent_id]
                )  # append (2*fov+1,2*fov+1,ch) to deque

                next_obss[agent_id] = np.concatenate(
                    self.frames[agent_id], axis=2
                ).astype(np.float32)  # (2*fov+1,2*fov+1,ch*n_frames)=(5,5,16)

                self.pos_frames[agent_id].append(
                    (self.env.reds[idx].pos[0] / self.env.config.grid_size,
                     self.env.reds[idx].pos[1] / self.env.config.grid_size)
                )  # append (2,) to deque

                next_poss[agent_id] = np.concatenate(
                    self.pos_frames[agent_id], axis=0
                ).astype(np.float32)  # (2*n_frames,)

            # Get padded next observations ndarray of all agent
            next_padded_obss = \
                make_padded_obs(
                    max_num_agents=self.env.config.max_num_red_agents,
                    obs_shape=self.obs_shape,
                    raw_obs=next_obss
                )  # (1,n,2*fov+1,2*fov+1,ch*n_frames)=(1,15,5,5,16)

            next_padded_poss = \
                make_padded_pos(
                    max_num_agents=self.env.config.max_num_red_agents,
                    pos_shape=self.pos_shape,
                    raw_pos=next_poss
                )  # (1,n,2*n_frames)=(1,15,8)

            # Get next_global_state
            self.global_frames.append(next_global_observation)
            # append (g,g,global_ch)

            next_global_state = np.concatenate(self.global_frames, axis=2).astype(np.float32)
            # (g,g,global_ch*global_n_frames)

            next_global_state = np.expand_dims(next_global_state, axis=0)
            # (1,g,g,global_ch*global_n_frames)

            # Get next mask for the padding
            next_mask = \
                make_mask(
                    alive_agents_ids=next_alive_agents_ids,
                    max_num_agents=self.env.config.max_num_red_agents
                )  # (1,n)

            next_attention_mask = \
                make_po_attention_mask(
                    alive_agents_ids=next_alive_agents_ids,
                    max_num_agents=self.env.config.max_num_red_agents,
                    agents=self.env.reds,
                    com=self.env.config.com
                )  # (1,n,n)

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

            global_r = np.expand_dims(
                np.array([reward], dtype=np.float32), axis=-1)  # append (1,1)
            global_done = np.expand_dims(
                np.array([done], dtype=bool), axis=-1)  # append (1,1)

            transition = (
                self.padded_obss,  # append (1,n,2*fov+1,2*fov+1,ch*n_frames)
                self.padded_poss,  # append (1,n,2*n_frames)
                padded_actions,  # append (1,n)
                padded_rewards,  # append (1,n)
                next_padded_obss,  # append (1,n,2*fov+1,2*fov+1,ch*n_frames)
                next_padded_poss,  # append (1,n,2*n_frames)
                padded_dones,  # append (1,n), bool
                global_r,  # append (1,1)
                global_done,  # append (1,1), bool
                self.mask,  # append (1,n), bool
                next_mask,  # append (1,n), bool
                self.attention_mask,  # append (1,n,n), bool
                next_attention_mask,  # append (1,n,n), bool
                self.global_state,  # append (1,g,g,global_ch*global_n_frames)
                next_global_state,  # append (1,g,g,global_ch*global_n_frames)
            )

            self.buffer.append(transition)

            if dones['all_dones']:
                # print(f'episode reward = {self.episode_return}')
                observations, global_observation = self.env.reset()
                self.reset_states(observations, global_observation)
            else:
                self.alive_agents_ids = next_alive_agents_ids
                self.padded_obss = next_padded_obss  # (1,15,5,5,16)
                self.padded_poss = next_padded_poss  # (1,15,8)
                self.mask = next_mask  # (1,15)
                self.attention_mask = next_attention_mask  # (1,15,15)

                self.global_state = next_global_state  # (1,g,g,global_ch*global_n_frames)

                self.step += 1

        transitions = self.buffer
        self.buffer = []

        return transitions, self.pid
