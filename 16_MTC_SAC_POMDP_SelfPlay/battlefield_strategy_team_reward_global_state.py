import gym
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pprint
import pickle
import tensorflow as tf
from pathlib import Path
from collections import deque

from agents_in_env import RED, BLUE
from config_dec_pomdp import Config
from generate_agents_in_env import generate_red_team, generate_blue_team

from rewards import get_consolidation_of_force_rewards, get_economy_of_force_rewards
from observations_global_state_dec_pomdp_self_play \
    import get_observations, get_global_observation, \
    get_global_blues_observation, get_blue_observations_po_0
from engage import engage_and_get_rewards, compute_engage_mask, get_dones

from generate_movies_dec_pomdp import MakeAnimation
from utils import compute_current_total_ef_and_force, count_alive_agents

from generate_movies_atention_map_dec_pomdp import MakeAnimation_AttentionMap

from models_global_state_mtc_dec_pomdp import MarlTransformerGlobalStateModel
from utils_transformer_mtc_dec_pomdp import make_blues_padded_obs, make_blues_padded_pos, \
    make_mask, make_po_attention_mask
from utils_gnn import get_alive_blue_agents_ids


class BattleFieldStrategy(gym.Env):
    def __init__(self):
        super(BattleFieldStrategy, self).__init__()

        self.config = Config()

        self.action_space = self.config.action_space

        self.observation_space = self.config.observation_space

        self.obs_shape = (2 * self.config.fov + 1,
                          2 * self.config.fov + 1,
                          self.config.observation_channels * self.config.n_frames)
        # (5,5,4*4), for blue agents

        self.pos_shape = (2 * self.config.n_frames,)  # (8,), for blue agents

        # Parameters TBD in reset()
        self.battlefield = None  # random_shape maze will be generated

        self.blues = None
        self.reds = None

        self.initial_blues_ef = None  # Initial (efficiency x Force) of the team (Max of total ef)
        self.initial_reds_ef = None

        self.initial_blues_force = None  # Initial force of the team (Max of total force)
        self.initial_reds_force = None

        self.initial_effective_blues_ef = None
        self.initial_effective_reds_ef = None

        self.initial_effective_blues_force = None
        self.initial_effective_reds_force = None

        self.step_count = None
        self.make_animation = None

        self.make_animation_attention_map = None

        #  build a neural network for blue_agents, then load weights
        # The shape of action_space & observation_space are identical to the one for red agents
        self.blue_policy, self.blue_alpha = self.build_blue_policy()
        # blue_alpha will not be used.

        ### The followings are reset in 'reset'
        self.blue_frames = None  # For each blue_agent @ t
        self.blue_pos_frames = None  # position of blue_agent @ t

        self.blue_global_frames = None  # Necessary only for build
        self.blue_global_state = None

        self.blue_alive_agents_ids = None  # For all agents, including dummy ones
        self.blue_padded_obss = None
        self.blue_padded_poss = None
        self.blue_padded_prev_actions = None  # TODO
        self.blue_mask = None  # alive_mask
        self.blue_attention_mask = None

    def build_blue_policy(self):  # copied from tester.main(), then modified.

        fov = self.config.fov

        """ global_state & feature """
        global_ch = self.config.global_observation_channels  # 6
        global_n_frames = self.config.global_n_frames

        global_state_shape = \
            (self.config.global_grid_size, self.config.global_grid_size,
             global_ch * global_n_frames)  # (15,15,24)

        global_state = np.ones(shape=global_state_shape)  # (15,15,24), for build
        global_state = np.expand_dims(global_state, axis=0)  # (1,15,15,24)

        """ agent observation """
        ch = self.config.observation_channels
        n_frames = self.config.n_frames

        obs_shape = (2 * fov + 1, 2 * fov + 1, ch * n_frames)  # (5,5,16)
        pos_shape = (2 * n_frames,)  # (8,)

        max_num_agents = self.config.max_num_blue_agents  # 15

        # Define alive_agents_ids & raw_obs for build
        alive_agents_ids = [0, 2, 3, 10]
        agent_obs = {}
        agent_pos = {}

        for i in alive_agents_ids:
            agent_id = 'blue_' + str(i)
            agent_obs[agent_id] = np.ones(obs_shape)
            agent_pos[agent_id] = np.ones(pos_shape) * i  # (8,)

        # Get padded_obs, padded_pos
        padded_obs = \
            make_blues_padded_obs(max_num_agents, obs_shape,
                                  agent_obs)  # (1,n,2*fov+1,2*fov+1,ch*n_frames)

        padded_pos = make_blues_padded_pos(max_num_agents, pos_shape, agent_pos)  # (1,n,2*n_frames)

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
        )  # (1,n,n), for build

        attention_mask = tf.cast(attention_mask, 'bool')

        # Make dummy policy and load learned weights
        dummy_policy = MarlTransformerGlobalStateModel(config=self.config)

        dummy_policy([[padded_obs, padded_pos], global_state], mask, attention_mask, training=False)

        # """ Use the followings for the test
        # Load model
        load_dir = Path(__file__).parent / 'blues_model'
        load_name = '/model_440000/'
        dummy_policy.load_weights(str(load_dir) + load_name)

        load_name = '/alpha_440000.npy'
        logalpha = np.load(str(load_dir) + load_name)
        logalpha = tf.Variable(logalpha)

        return dummy_policy, logalpha

    def reset(self):
        self.config.reset()  # Revise config for new episode

        self.battlefield = np.zeros((self.config.grid_size, self.config.grid_size))  # (g,g)

        """ Generate agent teams based on config """

        (self.blues, self.initial_blues_ef, self.initial_blues_force,
         self.initial_effective_blues_ef, self.initial_effective_blues_force) = \
            generate_blue_team(BLUE, self.config, self.battlefield)

        (self.reds, self.initial_reds_ef, self.initial_reds_force,
         self.initial_effective_reds_ef, self.initial_effective_reds_force) = \
            generate_red_team(RED, self.config, self.battlefield, self.blues)

        self.step_count = 0

        observations = get_observations(self)  # Get initial observations of agents (reds)

        global_observation = get_global_observation(self)

        blue_observations = \
            get_blue_observations_po_0(self)  # Get initial observations of agents (blues)

        global_blue_observation = get_global_blues_observation(self)

        """ Initialize blue observations """
        self.initialize_blue_obss_and_poss(blue_observations, global_blue_observation)

        self.make_animation = MakeAnimation(self)  # Animation generator

        self.make_animation_attention_map = MakeAnimation_AttentionMap(self)  # Attention map movie

        return observations, global_observation

    def initialize_blue_obss_and_poss(self, blue_observations, global_observation):
        self.blue_frames = {}
        self.blue_pos_frames = {}
        obss = {}
        poss = {}

        for blue in self.blues:
            # all reds are alive when reset

            self.blue_frames[blue.id] = deque([blue_observations[blue.id]] * self.config.n_frames,
                                              maxlen=self.config.n_frames)
            # [(2*fov+1,2*fov+1,channels),...], len=n_frames

            obss[blue.id] = np.concatenate(self.blue_frames[blue.id], axis=2).astype(np.float32)
            # (2*fov+1,2*fov+1,channels*n_frames)=(5,5,16)

            self.blue_pos_frames[blue.id] = \
                deque([(blue.pos[0] / self.config.grid_size,
                        blue.pos[1] / self.config.grid_size)] * self.config.n_frames,
                      maxlen=self.config.n_frames)
            # [(2,),..., len=n_frames

            poss[blue.id] = \
                np.concatenate(self.blue_pos_frames[blue.id], axis=0).astype(np.float32)
            # (2*n_frames,)

            # self.prev_actions[red.id] = 0

        # alive_agents_ids: list of alive agent id, [int,...], len=num_alive_agents
        self.blue_alive_agents_ids = get_alive_blue_agents_ids(env=self)

        # Get padded observations ndarray for all agents, including dead and dummy agents
        self.blue_padded_obss = \
            make_blues_padded_obs(max_num_agents=self.config.max_num_red_agents,
                                  obs_shape=self.obs_shape,
                                  raw_obs=obss)  # (1,n,2*fov+1,2*fov+1,ch*n_frames)=(1,15,5,5,16)

        self.blue_padded_poss = \
            make_blues_padded_pos(max_num_agents=self.config.max_num_red_agents,
                                  pos_shape=self.pos_shape,
                                  raw_pos=poss)  # (1,n,2*n_frames)=(1,15,8)

        # The tester does not use global state, but it is required for network instantiation.
        self.blue_global_frames = deque([global_observation] * self.config.global_n_frames,
                                        maxlen=self.config.global_n_frames)
        self.blue_global_frames = np.concatenate(self.blue_global_frames, axis=2).astype(np.float32)
        # (g,g,global_ch*global_n_frames)
        self.blue_global_state = np.expand_dims(self.blue_global_frames, axis=0)
        # (1,g,g,global_ch*global_n_frames)

        # Get alive mask for the padding
        self.blue_mask = make_mask(alive_agents_ids=self.blue_alive_agents_ids,
                                   max_num_agents=self.config.max_num_blue_agents)  # (1,n)

        # Get attention mask
        self.blue_attention_mask = \
            make_po_attention_mask(
                alive_agents_ids=self.blue_alive_agents_ids,
                max_num_agents=self.config.max_num_blue_agents,
                agents=self.blues,
                com=self.config.com
            )  # (1,n,n)

    def can_move(self, x, y):
        """
        no block: True
        block exits or outside of boundary: False
        """
        if (0 <= x < self.battlefield.shape[0]) and (0 <= y < self.battlefield.shape[1]):
            if self.battlefield[x, y] == 0:
                return True
            else:
                return False
        else:
            return False

    def move_reds(self, actions):
        """ action 0: NOT MOVE, 1: UP, 2: DOWN, 3: LEFT, 4: RIGHT """
        for red in self.reds:
            if red.alive:
                action = actions[red.id]

                if action == 1 and self.can_move(red.pos[0] - 1, red.pos[1]):
                    red.pos[0] -= 1  # UP

                elif action == 2 and self.can_move(red.pos[0] + 1, red.pos[1]):
                    red.pos[0] += 1  # DOWN

                elif action == 3 and self.can_move(red.pos[0], red.pos[1] - 1):
                    red.pos[1] -= 1  # LEFT

                elif action == 4 and self.can_move(red.pos[0], red.pos[1] + 1):
                    red.pos[1] += 1  # ROIGHT

                else:
                    pass  # Not move

                if (red.pos[0] < 0) or (red.pos[0] >= self.battlefield.shape[0]):
                    raise ValueError()

                if (red.pos[1] < 0) or (red.pos[1] >= self.battlefield.shape[1]):
                    raise ValueError()

    def move_blues(self, actions):
        for blue in self.blues:
            if blue.alive:
                action = actions[blue.id]

                if action == 1 and self.can_move(blue.pos[0] - 1, blue.pos[1]):
                    blue.pos[0] -= 1  # UP

                elif action == 2 and self.can_move(blue.pos[0] + 1, blue.pos[1]):
                    blue.pos[0] += 1  # DOWN

                elif action == 3 and self.can_move(blue.pos[0], blue.pos[1] - 1):
                    blue.pos[1] -= 1  # LEFT

                elif action == 4 and self.can_move(blue.pos[0], blue.pos[1] + 1):
                    blue.pos[1] += 1  # ROIGHT

                else:
                    pass  # Not move

                if (blue.pos[0] < 0) or (blue.pos[0] >= self.battlefield.shape[0]):
                    raise ValueError()

                if (blue.pos[1] < 0) or (blue.pos[1] >= self.battlefield.shape[1]):
                    raise ValueError()

    def initialize_step_rewards_and_dones(self, rewards, dones):
        """ for all alive agents, initialize by reward=-0.1, done=False """
        for red in self.reds:
            if red.alive:
                rewards[red.id] = -0.1  # Small negative reward in every time step
                dones[red.id] = False

        return rewards, dones

    def add_infos_of_reds(self, infos):
        for red in self.reds:
            infos[red.id] = {'time': np.round(self.step_count * self.config.dt, 2),
                             'type': red.type,
                             'efficiency': red.efficiency,
                             'ef': red.ef,
                             'force': red.force,
                             'alive': red.alive
                             }

        return infos

    def is_all_dones(self, dones, infos):
        """
        End of episode ?
        """
        reds_alive_list = []
        for red in self.reds:
            reds_alive_list.append(red.alive)

        blues_alive_list = []
        for blue in self.blues:
            blues_alive_list.append(blue.alive)

        reds_survive = any(reds_alive_list)
        blues_survive = any(blues_alive_list)

        if reds_survive and blues_survive:
            dones['all_dones'] = False
        else:
            dones['all_dones'] = True

            if reds_survive and not blues_survive:
                infos['win'] = 'reds'

                _, _, remaining_effective_ef_reds, remaining_effective_force_reds = \
                    compute_current_total_ef_and_force(self.reds)

                infos['remaining_effective_ef_reds'] = remaining_effective_ef_reds
                infos['remaining_effective_force_reds'] = remaining_effective_force_reds
                infos['remaining_effective_ef_blues'] = 0
                infos['remaining_effective_force_blues'] = 0

            elif not reds_survive and blues_survive:
                infos['win'] = 'blues'

                _, _, remaining_effective_ef_blues, remaining_effective_force_blues = \
                    compute_current_total_ef_and_force(self.blues)

                infos['remaining_effective_ef_reds'] = 0
                infos['remaining_effective_force_reds'] = 0
                infos['remaining_effective_ef_blues'] = remaining_effective_ef_blues
                infos['remaining_effective_force_blues'] = remaining_effective_force_blues

            else:
                infos['win'] = 'draw'

                _, _, remaining_effective_ef_reds, remaining_effective_force_reds = \
                    compute_current_total_ef_and_force(self.reds)
                _, _, remaining_effective_ef_blues, remaining_effective_force_blues = \
                    compute_current_total_ef_and_force(self.blues)

                infos['remaining_effective_ef_reds'] = remaining_effective_ef_reds
                infos['remaining_effective_force_reds'] = remaining_effective_force_reds
                infos['remaining_effective_ef_blues'] = remaining_effective_ef_blues
                infos['remaining_effective_force_blues'] = remaining_effective_force_blues

            for red in self.reds:
                dones[red.id] = True

        return dones, infos

    def step(self, actions):
        """
         1. Based on the actions, alive agents (reds & blue) move to the new cell.
         2. Get additional reward for consolidation of power (before engage)
         3. Engage (1-step Lanchester) at each cell and update the ef and force of agents in the
            cell. All agents joined to the engagement get rewards based on the before and after the
            agents and cell status
         4. Get (next) observations after engagement
         5. Evaluate dones and agent aliveness after the engagement
        """

        self.step_count += 1

        rewards = {}
        dones = {}
        infos = {}

        num_engaging_reds = 0
        num_engaging_blues = 0
        engaging_force_reds = 0
        engaging_force_blues = 0

        reward = self.energy_consumption()  # team reward at timestep, small negative value

        rewards, dones = self.initialize_step_rewards_and_dones(rewards, dones)

        reds_num_before, reds_force_before, blues_num_before, blues_force_before = \
            self.get_team_params()

        """ Get blue_actions """
        blue_actions = self.get_blue_actions()  # dict

        # 1. move to the new cell
        self.move_reds(actions)
        self.move_blues(blue_actions)

        # 2. Get additional rewards before engagement.
        # Use this additional rewards with get_rewards.
        # rewards = get_consolidation_of_force_rewards(self, rewards)
        # rewards = get_economy_of_force_rewards(self, rewards)

        # 3. Engage (1-step Lanchester) and update the ef and force of agents in the cell,
        #    then get rewards based on the before and after agents and cell status
        engage_mask, _, _, _, _ = compute_engage_mask(self)  # (grid_size,grid_size)
        (x1, y1) = np.where(engage_mask == 1)

        if len(x1) > 0:
            # Before engagements
            num_engaging_reds, engaging_force_reds = \
                self.compute_force_to_engage(x1, y1, self.reds)

            num_engaging_blues, engaging_force_blues = \
                self.compute_force_to_engage(x1, y1, self.blues)

            # agent ef, force update & get rewards, when engage
            rewards, infos = engage_and_get_rewards(self, x1, y1, rewards, infos)

        # 4. Evaluate dones and agent aliveness after engagement
        if len(x1) > 0:
            dones, infos = get_dones(self, x1, y1, dones, infos)

        dones, infos = self.is_all_dones(dones, infos)

        infos = self.add_infos_of_reds(infos)

        check_code_agent_alive(self.reds)
        check_code_agent_alive(self.blues)

        # Get team reward after engagement
        if len(x1) > 0:
            reds_num_after, reds_force_after, blues_num_after, blues_force_after = \
                self.get_team_params()

        else:
            reds_num_after = reds_num_before
            reds_force_after = reds_force_before
            blues_num_after = blues_num_before
            blues_force_after = blues_force_before

        """ reward @ every timestep """
        reward += self.average_force_to_engage_reward(
            blues_force_after, blues_force_before, reds_force_after, reds_force_before,
            engaging_force_reds, engaging_force_blues, x1, y1, actions
        )

        reward += self.enemy_attrition_reward(blues_num_after, blues_num_before)

        """ Is team done ? """
        done = self.get_team_done(dones)

        """ reward @ done (end of episode) """
        if done:
            # For amount of force
            reward += self.end_of_episode_reward(reds_force_after,
                                                 self.initial_effective_reds_force,
                                                 blues_force_after,
                                                 self.initial_effective_blues_force, infos)

        # 5. Get (next) observations after engagement, red_agents
        observations = get_observations(self)

        # Update observations and positions of blue_agents
        self.update_blues_states()

        # Get (next) global observations
        global_observation = get_global_observation(self)

        return observations, rewards, dones, infos, reward, done, global_observation

    def update_blues_states(self):
        """ Update blue agents obss & poss after engagement """

        next_blue_observations = get_blue_observations_po_0(self)  # dict

        next_alive_blue_agents_ids = get_alive_blue_agents_ids(env=self)  # list

        ### For alive agents in env
        next_blue_obss = {}
        next_blue_poss = {}

        for idx in next_alive_blue_agents_ids:
            agent_id = 'blue_' + str(idx)

            self.blue_frames[agent_id].append(
                next_blue_observations[agent_id]
            )  # append (2*fov+1,2*fov+1,ch) to deque; update obs_frames

            next_blue_obss[agent_id] = np.concatenate(
                self.blue_frames[agent_id], axis=2
            ).astype(np.float32)  # (2*fov+1,2*fov+1,ch*n_frames)=(5,5,16)

            self.blue_pos_frames[agent_id].append(
                (self.blues[idx].pos[0] / self.config.grid_size,
                 self.blues[idx].pos[1] / self.config.grid_size)
            )  # append (2,) to deque; update pos_frames

            next_blue_poss[agent_id] = np.concatenate(
                self.blue_pos_frames[agent_id], axis=0
            ).astype(np.float32)  # (2*n_frames,)

        # Get padded next observations ndarray of all agent
        next_blue_padded_obss = \
            make_blues_padded_obs(
                max_num_agents=self.config.max_num_blue_agents,
                obs_shape=self.obs_shape,
                raw_obs=next_blue_obss
            )  # (1,n,2*fov+1,2*fov+1,ch*n_frames)=(1,15,5,5,16)

        next_blue_padded_poss = \
            make_blues_padded_pos(
                max_num_agents=self.config.max_num_blue_agents,
                pos_shape=self.pos_shape,
                raw_pos=next_blue_poss
            )  # (1,n,2*n_frames)=(1,15,8)

        # The blue does not use global state except build. I omit updating the blue_global_state.

        # Get next mask for the padding
        next_blue_mask = \
            make_mask(
                alive_agents_ids=next_alive_blue_agents_ids,
                max_num_agents=self.config.max_num_blue_agents
            )  # (1,n)

        next_blue_attention_mask = \
            make_po_attention_mask(
                alive_agents_ids=next_alive_blue_agents_ids,
                max_num_agents=self.config.max_num_blue_agents,
                agents=self.blues,
                com=self.config.com
            )  # (1,n,n)

        # update self (global state is not necessary)
        self.blue_alive_agents_ids = next_alive_blue_agents_ids
        self.blue_padded_obss = next_blue_padded_obss
        self.blue_padded_poss = next_blue_padded_poss
        self.blue_padded_prev_actions = None  # TODO
        self.blue_mask = next_blue_mask
        self.blue_attention_mask = next_blue_attention_mask

    def get_blue_actions(self):
        """ get blues actions as dictionary """
        blue_acts, _ = \
            self.blue_policy.sample_actions([self.blue_padded_obss, self.blue_padded_poss],
                                            self.blue_mask, self.blue_attention_mask,
                                            training=False)  # blue_acts: (1,n)

        # get alive_agents & all agents actions. action=0 <- do nothing
        actions = {}  # For alive agents

        for idx in self.blue_alive_agents_ids:
            agent_id = 'blue_' + str(idx)
            actions[agent_id] = blue_acts[0, idx]

        return actions

    def render(self, mode=None):
        pass

    def get_team_params(self):

        blues_num = 0
        blues_force = 0
        reds_num = 0
        reds_force = 0

        for blue in self.blues:
            if blue.alive:
                blues_num += 1
                blues_force += blue.effective_force

        for red in self.reds:
            if red.alive:
                reds_num += 1
                reds_force += red.effective_force

        return reds_num, reds_force, blues_num, blues_force

    @staticmethod
    def symlog(x):
        # scaling function of reward (adopted from Dreamer-V3)
        y = np.sign(x) * np.log(np.abs(x) + 1)
        return y

    def energy_consumption(self):
        x = -0.5
        return self.symlog(x)

    def average_force_to_engage_reward(self, blues_force_after, blues_force_before,
                                       reds_force_after, reds_force_before,
                                       engaging_force_reds, engaging_force_blues, x1, y1, actions):

        # reward when engagement occurs.
        if (blues_force_after != blues_force_before) and (reds_force_after != reds_force_before):
            # reward when engagement occurs.

            if reds_force_after >= blues_force_after:
                if engaging_force_reds >= engaging_force_blues:
                    r = engaging_force_reds / engaging_force_blues
                    r = 2.0 / np.pi * np.arctan(r - 1)  # 0<=r<1
                    reward = 0.7 + 0.8 * r / len(x1)
                else:
                    reward = 0.6

            else:
                if engaging_force_reds >= engaging_force_blues:
                    r = engaging_force_reds / engaging_force_blues
                    r = 2.0 / np.pi * np.arctan(r - 1)  # 0<=r<1
                    reward = 0.4 + 0.1 * r / len(x1)
                else:
                    reward = 0.3

        else:
            # reward when no-engagement and search enemy
            num_reds_alive, num_reds_cluster = self.count_group()
            r1 = num_reds_cluster / num_reds_alive  # 0<r1<=1

            num_move_reds = self.count_move_reds(actions)
            r2 = num_move_reds / num_reds_alive  # 0<=r2<=1

            if (r1 > 1.) or (r2 > 1.):
                raise ValueError()

            r = (r1 + r2) / 2.0  # 0<r<=1

            reward = 0.2 * r

        return self.symlog(reward)

    def count_move_reds(self, actions):
        num_move_reds = 0
        for red in self.reds:
            if red.alive:
                if actions[red.id] != 0:
                    num_move_reds += 1

        return num_move_reds

    def count_group(self):
        """ reds の alive数とcluster数をカウント"""
        alive_count = 0
        x = []
        y = []
        for red in self.reds:
            if red.alive:
                alive_count += 1
                x.append(red.pos[0])
                y.append(red.pos[1])

        x = np.array(x)
        y = np.array(y)

        group_count = 0
        for i in range(self.config.grid_size):
            for j in range(self.config.grid_size):
                num = np.sum(np.where(x == i, 1, 0) * np.where(y == j, 1, 0))
                if num > 0:
                    group_count += 1

        return alive_count, group_count

    def enemy_attrition_reward(self, blues_num_after, blues_num_before):
        c2 = 2.0

        reward = blues_num_before - blues_num_after

        return self.symlog(c2 * reward)

    def compute_force_to_engage(self, x1, y1, agents):
        counter = 0
        engaging_force = 0
        total_force = 0

        for agent in agents:
            if agent.alive:
                x = agent.pos[0]
                y = agent.pos[1]
                total_force += agent.effective_force

                if (x in x1) and (y in y1):
                    counter += 1
                    engaging_force += agent.effective_force

        return counter, engaging_force

    def end_of_episode_reward(self, reds_force_after, R0, blues_force_after, B0, infos):
        if infos["win"] == "reds":
            reward = 10

        else:
            reward = -10

        return self.symlog(reward)

    @staticmethod
    def get_team_done(dones):
        if dones["all_dones"]:
            done = True
        else:
            done = False

        return done


def get_results_for_summary(env, agent, summary):
    """ Called from summary_of_episode """
    summary[agent.id] = {'pos': agent.pos,
                         'ef': np.round(agent.ef, 1),
                         'force': np.round(agent.force, 1),
                         'alive': agent.alive}
    return summary


def summary_of_episode(env, dones, infos):
    summary = {}

    for red in env.reds:
        summary = get_results_for_summary(env, red, summary)

    for blue in env.blues:
        summary = get_results_for_summary(env, blue, summary)

    tf = np.round(env.config.dt * env.step_count, 2)

    print(f'@ {tf} sec: ', end='')

    if dones['all_dones']:
        print('all dones:', dones['all_dones'], end='')
        print(', win is', infos['win'])
    else:
        print('------------- time up')

    pprint.pprint(summary)


def count_wins(infos, total_return,
               returns_reds_win, returns_blues_win, returns_draw, returns_no_contest,
               remaining_effective_ef_reds, remaining_effective_force_reds,
               remaining_effective_ef_blues, remaining_effective_force_blues, env):
    initial_reds_list = []
    initial_blues_list = []

    if 'win' in infos:
        if infos['win'] == 'reds':
            returns_reds_win.append(total_return)
        elif infos['win'] == 'blues':
            returns_blues_win.append(total_return)

            for red in env.reds:
                initial_reds_list.append([red.type, red.initial_force, red.efficiency])

            for blue in env.blues:
                initial_blues_list.append([blue.type, blue.initial_force, blue.efficiency])

        else:
            returns_draw.append(total_return)
    else:
        returns_no_contest.append(total_return)

    remaining_effective_ef_reds.append(infos['remaining_effective_ef_reds'])
    remaining_effective_force_reds.append(infos['remaining_effective_force_reds'])
    remaining_effective_ef_blues.append(infos['remaining_effective_ef_blues'])
    remaining_effective_force_blues.append(infos['remaining_effective_force_blues'])

    return (returns_reds_win, returns_blues_win, returns_draw, returns_no_contest,
            remaining_effective_ef_reds, remaining_effective_force_reds,
            remaining_effective_ef_blues, remaining_effective_force_blues,
            initial_reds_list, initial_blues_list)


def summary_of_whole_episodes(max_episodes, returns_reds_win,
                              returns_blues_win, returns_draw, returns_no_contest):
    num_reds_win = len(returns_reds_win)
    num_blues_win = len(returns_blues_win)
    num_draws = len(returns_draw)
    num_no_contest = len(returns_no_contest)

    print('\n==============================================================')
    print(f'reds wins:{num_reds_win} over {max_episodes} episodes')
    print(f'blues wins:{num_blues_win} over {max_episodes} episodes')
    print(f'draws:{num_draws} over {max_episodes} episodes')
    print(f'no_contest:{num_no_contest} over {max_episodes} episodes')
    print('\n')

    if num_reds_win > 0:
        print(f'reds wins return: mean={np.round(np.mean(returns_reds_win), 2)}, '
              f'std={np.round(np.std(returns_reds_win), 2)}')
    else:
        print('No reds win')

    if num_blues_win > 0:
        print(f'blues wins return: mean={np.round(np.mean(returns_blues_win), 2)}, '
              f'std={np.round(np.std(returns_blues_win), 2)}')
    else:
        print('No blues win')

    if num_draws > 0:
        print(f'blues wins return: mean={np.mean(returns_draw)}, '
              f'std={np.std(returns_draw)}')
    else:
        print('No draw')

    print(f'ratio of no_contest: {np.round(num_no_contest / max_episodes, 2)}')

    return num_reds_win, num_blues_win, num_draws, num_no_contest


def draw_distributions(agent_color, return_agents_win, num_agents_win, filedir, time_stamp):
    if agent_color == 'reds':
        hist_color = 'red'
    elif agent_color == 'blues':
        hist_color = 'blue'
    else:
        raise NotImplementedError()

    plt.hist(return_agents_win, bins=10, color=hist_color)
    plt.title(f'distributing of {agent_color} returns over {num_agents_win} episodes')

    filename = os.path.join(filedir,
                            str(agent_color) + '_wins_' + time_stamp + '.png')
    plt.savefig(filename)
    # plt.show()
    plt.close()


def save_returns_list(returns_reds_win, returns_blues_win, returns_draw, returns_no_contest,
                      filedir, time_stamp):
    filename = os.path.join(filedir, 'returns_reds_win' + time_stamp + '.txt')
    f = open(filename, 'wb')
    pickle.dump(returns_reds_win, f)

    # f = open(filename, "rb")
    # returns_reds_win = pickle.load(f)

    filename = os.path.join(filedir, 'returns_blues_win' + time_stamp + '.txt')
    f = open(filename, 'wb')
    pickle.dump(returns_blues_win, f)

    # f = open(filename, "rb")
    # returns_blues_win = pickle.load(f)

    filename = os.path.join(filedir, 'returns_draw' + time_stamp + '.txt')
    f = open(filename, 'wb')
    pickle.dump(returns_draw, f)

    # f = open(filename, "rb")
    # returns_draw = pickle.load(f)

    filename = os.path.join(filedir, 'returns_no_conest' + time_stamp + '.txt')
    f = open(filename, 'wb')
    pickle.dump(returns_no_contest, f)

    # f = open(filename, "rb")
    # returns_no_contest = pickle.load(f)

    # print(returns_reds_win, returns_blues_win, returns_draw, returns_no_contest)


def check_code_agent_alive(agents):
    """
    agents = env.reds or env.blues
    """
    for agent in agents:
        if agent.alive:
            criteria_1 = (agent.force > agent.threshold)
            criteria_2 = (agent.effective_force > 0)

            if not (criteria_1 and criteria_2):
                raise ValueError()

        else:
            criteria_1 = (agent.force <= agent.threshold)
            criteria_2 = (agent.effective_force <= 0)

            if not (criteria_1 and criteria_2):
                raise ValueError()
