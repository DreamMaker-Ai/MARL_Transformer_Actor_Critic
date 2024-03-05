import ray
import numpy as np

from collections import deque

from models_global_state import MarlTransformerGlobalStateModel
from battlefield_strategy_team_reward_global_state import BattleFieldStrategy
from utils_gnn import get_alive_agents_ids
from utils_transformer import make_id_mask as make_mask


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

        self.obs_shape = (self.env.config.grid_size,
                          self.env.config.grid_size,
                          self.env.config.observation_channels * self.n_frames)

        # Define local buffer
        self.buffer = []

        # Initialize environment
        ### The followings are reset in 'reset_states'
        self.frames = None  # For each agent in env
        self.prev_actions = None

        self.alive_agents_ids = None  # For all agents, including dummy ones
        self.padded_obss = None
        self.padded_prev_actions = None
        self.masks = None

        self.global_frames = None
        self.global_state = None

        self.episode_return = None
        self.step = None

        ### Initialize above Nones
        observations, global_observation = self.env.reset()
        self.reset_states(observations, global_observation)

        self.mtc([self.padded_obss, self.global_state], self.masks, training=True)  # build

    def reset_states(self, observations, global_observation):
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
        self.padded_obss = []
        self.prev_actions = {}

        # alive_agents_ids: list of alive agent id, [int,...], len=num_alive_agents
        self.alive_agents_ids = get_alive_agents_ids(env=self.env)

        # observations list
        for idx in range(self.env.config.max_num_red_agents):
            agent_id = 'red_' + str(idx)

            if idx in self.alive_agents_ids:
                self.frames[agent_id] = deque([observations[agent_id]] * self.n_frames,
                                              maxlen=self.n_frames)
                # [(g,g,ch),...], len=n_frames

                obs = np.concatenate(self.frames[agent_id], axis=2).astype(np.float32)
                # (g,g,ch*n_frames)
            else:
                obs = np.zeros(self.obs_shape)  # (g,g,ch*n_frames)

            obs = np.expand_dims(obs, axis=0)  # add batch_dim, (1,g,g,ch*n_frames)

            self.padded_obss.append(obs)  # [(1,g,g,ch*n_frames),...], len=n

        # global state
        self.global_frames = deque(
            [global_observation] * self.global_n_frames,
            maxlen=self.global_n_frames
        )

        global_frames = np.concatenate(self.global_frames, axis=2).astype(np.float32)

        self.global_state = \
            np.expand_dims(global_frames, axis=0)  # (1,g,g,global_ch*global_n_frames)

        # Get mask for the padding
        self.masks = make_mask(alive_agents_ids=self.alive_agents_ids,
                               max_num_agents=self.env.config.max_num_red_agents)
        # [(1,1,n),...], len=n

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
            
            transitions = [transition, ...], list of experience, len=b
            
            :transition: 1 experience of multi-agent = (
                self.padded_obss,  # [(1,g,g,ch*n_frames),...], len=n
                padded_actions,  # [(1,),...], len=n
                agents_rewards,  # [(1,),...], len=n
                next_padded_obss,  # [(1,g,g,ch*n_frames),...], len=n
                agents_dones,  # [(1,),...], len=n, bool
                global_r,  # team_r; (1,1)
                global_done,  # team_done; (1,1), bool
                self.masks,  # [(1,1,n),...], len=n, bool
                next_masks,  # [(1,1,n),...], len=n, bool
                self.global_state,  # (1,g,g,global_ch*global_n_frames)
                next_global_state,  # (1,g,g,global_ch*global_n_frames)
            )
        """

        trajectory = self._rollout()

        return trajectory

    def _rollout(self):
        """
        Rolloutにより、t_start<=t<t_max 間(batch_size間)の transition (experience) を取得
        """

        # Rollout実施
        for i in range(self.batch_size):
            acts, scores = \
                self.mtc.sample_actions(self.padded_obss, self.masks, training=False)
            # acts: [(1,1),...], len=n, int32
            # [[score1, score2]:[(1,num_heads,n),...],[(1,num_heads,n),...]]

            # get alive_agents & all agents actions.
            # * padded_actions: action=-1 for dead or dummy agents for
            #       utilizing tf.one_hot(-1) is zero vector
            # * actions['red_a'], a=alive agent id
            actions = {}  # For alive agents
            padded_actions = []

            for idx in range(self.env.config.max_num_red_agents):
                if idx in self.alive_agents_ids:
                    agent_id = 'red_' + str(idx)
                    actions[agent_id] = acts[idx][0,0]  # int32

                    padded_actions.append(np.array([actions[agent_id]]))  # append (1,)
                else:
                    padded_actions.append(np.array([-1]))  # append (1,)

            # One step of Lanchester simulation, for alive agents in env
            next_obserations, rewards, dones, infos, reward, done, next_global_observation = \
                self.env.step(actions)

            # Make next_agents_states, next_agents_adjs, and next_alive_agents_ids,
            # including dummy ones
            next_alive_agents_ids = get_alive_agents_ids(env=self.env)

            ### For alive agents in env
            next_padded_obss = []

            for idx in range(self.env.config.max_num_red_agents):
                agent_id = 'red_' + str(idx)

                if idx in next_alive_agents_ids:
                    self.frames[agent_id].append(next_obserations[agent_id])
                    # append (g,g,ch) to deque

                    next_obs = np.concatenate(self.frames[agent_id], axis=2).astype(np.float32)
                    # (g,g,ch*n_frames)
                else:
                    next_obs = np.zeros(self.obs_shape)  # (g,g,ch*n_frames)

                next_obs = np.expand_dims(next_obs, axis=0)  # (1,g,g,ch*n_frames)
                next_padded_obss.append(next_obs)  # [(1,g,g,ch*n_frames),...], len=n

            # Get next_global_state
            self.global_frames.append(next_global_observation)
            # append (g,g,global_ch)

            next_global_state = np.concatenate(self.global_frames, axis=2).astype(np.float32)
            # (g,g,global_ch*global_n_frames)

            next_global_state = np.expand_dims(next_global_state, axis=0)
            # (1,g,g,global_ch*global_n_frames)

            # Get next mask for the padding
            next_masks = \
                make_mask(
                    alive_agents_ids=next_alive_agents_ids,
                    max_num_agents=self.env.config.max_num_red_agents
                )  # [(1,1,n),...], len=n

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
                    agents_rewards.append(np.array([float(rewards[agent_id])]))  # append (1,)
                    agents_dones.append(np.array([dones[agent_id]]))
                else:
                    agents_rewards.append(np.array([0.0]))  # append (1,)
                    agents_dones.append(np.array([True]))

            # if len(agents_rewards) != self.env.config.max_num_red_agents:
            #     raise ValueError()

            # if len(agents_dones) != self.env.config.max_num_red_agents:
            #     raise ValueError()

            # Update episode return
            self.episode_return += np.sum(agents_rewards)

            # alive_agents_ids = np.array(self.alive_agents_ids, dtype=object)  # (a,), object
            # alive_agents_ids = np.expand_dims(alive_agents_ids, axis=0)  # (1,a)

            global_r = np.expand_dims(
                np.array([reward], dtype=np.float32), axis=-1)  # append (1,1)
            global_done = np.expand_dims(
                np.array([done], dtype=bool), axis=-1)  # append (1,1)

            # 1 experience of multi-agent
            transition = (
                self.padded_obss,  # [(1,g,g,ch*n_frames),...], len=n
                padded_actions,  # [(1,),...]
                agents_rewards,  # [(1,),...]
                next_padded_obss,  # [(1,g,g,ch*n_frames),...]
                agents_dones,  # [(1,),...], bool
                global_r,  # team_r; (1,1)
                global_done,  # team_done; (1,1), bool
                self.masks,  # [(1,1,n),...], bool
                next_masks,  # [(1,1,n),...], bool
                self.global_state,  # (1,g,g,global_ch*global_n_frames)
                next_global_state,  # (1,g,g,global_ch*global_n_frames)
            )

            self.buffer.append(transition)

            if dones['all_dones']:
                # print(f'episode reward = {self.episode_return}')
                observations, global_observation = self.env.reset()
                self.reset_states(observations, global_observation)
            else:
                self.alive_agents_ids = next_alive_agents_ids
                self.padded_obss = next_padded_obss
                self.masks = next_masks

                self.global_state = next_global_state  # (1,g,g,global_ch*global_n_frames)

                self.step += 1

        transitions = self.buffer  # len=b (= batch_size = sequence len)
        self.buffer = []

        return transitions, self.pid
