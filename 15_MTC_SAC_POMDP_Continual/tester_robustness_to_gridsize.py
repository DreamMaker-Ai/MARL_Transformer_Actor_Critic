import os.path
import random
import json
from pathlib import Path

import ray
import gym
import tensorflow as tf
import numpy as np
from collections import deque

from battlefield_strategy_team_reward_global_state import BattleFieldStrategy

from models_global_state_mtc_dec_pomdp import MarlTransformerGlobalStateModel
from global_models_dec_pomdp import GlobalCNNModel

from utils_gnn import get_alive_agents_ids
from utils_transformer_mtc_dec_pomdp import make_mask, make_po_attention_mask, \
    make_padded_obs, make_padded_pos

import matplotlib.pyplot as plt

"""
Tester does not use global state, but it is required for network instantiation.
"""


def who_wins(result_red, result_blue):
    num_alive_reds = result_red['num_alive_agent']
    num_alive_blues = result_blue['num_alive_agent']

    if (num_alive_reds <= 0) & (num_alive_blues <= 0):
        winner = 'draw'

    elif num_alive_reds <= 0:
        winner = 'blue_win'

    elif num_alive_blues <= 0:
        winner = 'red_win'

    else:
        winner = 'no_contest'

    return winner


def summarize_agent_result(agents):
    num_platoon = 0
    num_company = 0
    num_alive_platoon = 0
    num_alive_company = 0
    remaining_effective_force = 0
    initial_effective_force = 0

    for agent in agents:
        if agent.type == 'platoon':
            num_platoon += 1
        else:
            num_company += 1

        if agent.alive:
            if agent.type == 'platoon':
                num_alive_platoon += 1
            else:
                num_alive_company += 1

            remaining_effective_force += agent.effective_force

        initial_effective_force += agent.initial_effective_force

    num_initial_agents = num_platoon + num_company
    num_alive = num_alive_platoon + num_alive_company

    result = {}

    result['num_initial_agent'] = num_initial_agents
    result['num_initial_platoon'] = num_platoon
    result['num_initial_company'] = num_company

    result['num_alive_agent'] = num_alive
    result['num_alive_platoon'] = num_alive_platoon
    result['num_alive_company'] = num_alive_company

    result['initial_effective_force'] = initial_effective_force
    result['remaining_effective_force'] = remaining_effective_force

    return result


def summarize_episode_results(results, result_red, result_blue, winner):
    # For reds
    results['alive_reds_ratio'].append(
        result_red['num_alive_agent'] / result_red['num_initial_agent']
    )

    results['alive_red_platoon'].append(result_red['num_alive_platoon'])
    results['alive_red_company'].append(result_red['num_alive_company'])

    results['initial_red_platoon'].append(result_red['num_initial_platoon'])
    results['initial_red_company'].append(result_red['num_initial_company'])

    results['remaining_red_effective_force_ratio'].append(
        result_red['remaining_effective_force'] / result_red['initial_effective_force']
    )

    # For blues
    results['alive_blues_ratio'].append(
        result_blue['num_alive_agent'] / result_blue['num_initial_agent']
    )

    results['alive_blue_platoon'].append(result_blue['num_alive_platoon'])
    results['alive_blue_company'].append(result_blue['num_alive_company'])

    results['initial_blue_platoon'].append(result_blue['num_initial_platoon'])
    results['initial_blue_company'].append(result_blue['num_initial_company'])

    results['remaining_blue_effective_force_ratio'].append(
        result_blue['remaining_effective_force'] / result_blue['initial_effective_force']
    )

    results['winner'].append(winner)

    return results


def summarize_results(results):
    result = {}

    result['episode_rewards'] = np.mean(results['episode_rewards'])
    result['episode_lens'] = np.mean(results['episode_lens'])
    result['episode_team_return'] = np.mean(results['episode_team_return'])

    result['num_alive_reds_ratio'] = np.mean(results['alive_reds_ratio'])
    result['num_alive_red_platoon'] = np.mean(results['alive_red_platoon'])
    result['num_alive_red_company'] = np.mean(results['alive_red_company'])
    result['remaining_red_effective_force_ratio'] = \
        np.mean(results['remaining_red_effective_force_ratio'])

    result['num_initial_red_platoon'] = np.mean(results['initial_red_platoon'])
    result['num_initial_red_company'] = np.mean(results['initial_red_company'])

    result['num_alive_blues_ratio'] = np.mean(results['alive_blues_ratio'])
    result['num_alive_blue_platoon'] = np.mean(results['alive_blue_platoon'])
    result['num_alive_blue_company'] = np.mean(results['alive_blue_company'])
    result['remaining_blue_effective_force_ratio'] = \
        np.mean(results['remaining_blue_effective_force_ratio'])

    result['num_initial_blue_platoon'] = np.mean(results['initial_blue_platoon'])
    result['num_initial_blue_company'] = np.mean(results['initial_blue_company'])

    result['num_red_win'] = results['winner'].count('red_win')
    result['num_blue_win'] = results['winner'].count('blue_win')
    result['draw'] = results['winner'].count('draw')
    result['no_contest'] = results['winner'].count('no_contest')

    return result


@ray.remote
# @ray.remote(num_cpus=1, num_gpus=0)
class Tester:
    def __init__(self):
        # Make a copy of environment
        self.env = BattleFieldStrategy()
        self.action_space_dim = self.env.action_space.n
        self.n_frames = self.env.config.n_frames
        self.epsilon = 0.0

        self.obs_shape = (2 * self.env.config.fov + 1,
                          2 * self.env.config.fov + 1,
                          self.env.config.observation_channels * self.n_frames)  # (5,5,4*4)

        self.pos_shape = (2 * self.env.config.n_frames,)  # (8,)

        self.global_n_frames = self.env.config.global_n_frames  # frame stacks of global states

        # Make an actor-critic network
        self.policy = MarlTransformerGlobalStateModel(config=self.env.config)

        # Initialize environment
        ### The followings are reset in 'reset_states'
        self.frames = None  # For each agent in env
        self.pos_frames = None  # position of agent
        self.prev_actions = None

        self.global_frames = None  # For frame stack og global states

        self.alive_agents_ids = None  # For all agents, including dummy ones
        self.padded_obss = None
        self.padded_poss = None
        self.padded_prev_actions = None  # TODO
        self.mask = None  # alive_mask
        self.attention_mask = None

        ### Tester does not use global state, but it is required for network instantiation.
        self.global_state = None

        # self.episode_reward = None
        self.step = None

        ### Initialize above Nones
        observations, global_state = self.env.reset()
        self.reset_states(observations, global_state)

        # make_animation
        if self.env.config.make_animation:
            self.env.make_animation.add_frame(self.env)

        # For saving best model
        self.num_max_win = -1
        self.max_return = -1000000

    def reset_states(self, observations, global_state):
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
        self.prev_actions = {}  # TODO

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
            # (2*n_frames,)

            # self.prev_actions[red.id] = 0

        # alive_agents_ids: list of alive agent id, [int,...], len=num_alive_agents
        self.alive_agents_ids = get_alive_agents_ids(env=self.env)

        # Get padded observations ndarray for all agents, including dead and dummy agents
        self.padded_obss = \
            make_padded_obs(max_num_agents=self.env.config.max_num_red_agents,
                            obs_shape=self.obs_shape,
                            raw_obs=obss)  # (1,n,2*fov+1,2*fov+1,ch*n_frames)=(1,15,5,5,16)

        self.padded_poss = \
            make_padded_pos(max_num_agents=self.env.config.max_num_red_agents,
                            pos_shape=self.pos_shape,
                            raw_pos=poss)  # (1,n,2*n_frames)=(1,15,8)

        # The tester does not use global state, but it is required for network instantiation.
        self.global_frames = deque([global_state] * self.global_n_frames,
                                   maxlen=self.global_n_frames)
        self.global_frames = np.concatenate(self.global_frames, axis=2).astype(np.float32)
        # (g,g,global_ch*global_n_frames)
        self.global_state = np.expand_dims(self.global_frames, axis=0)
        # (1,g,g,global_ch*global_n_frames)

        # Get alive mask for the padding
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

        # Build policy
        self.policy([[self.padded_obss, self.padded_poss], self.global_state],
                    self.mask, self.attention_mask, training=False)

        # reset episode variables
        self.step = 0

    def initialize_results(self):
        results = {}

        results['episode_rewards'] = []
        results['episode_lens'] = []

        results['episode_team_return'] = []

        results['alive_red_platoon'] = []
        results['alive_red_company'] = []
        results['alive_reds_ratio'] = []
        results['remaining_red_effective_force_ratio'] = []

        results['initial_red_platoon'] = []
        results['initial_red_company'] = []

        results['alive_blue_platoon'] = []
        results['alive_blue_company'] = []
        results['alive_blues_ratio'] = []
        results['remaining_blue_effective_force_ratio'] = []

        results['initial_blue_platoon'] = []
        results['initial_blue_company'] = []

        results['winner'] = []

        return results

    def test_play(self, current_weights):
        # 重みを更新
        self.policy.set_weights(weights=current_weights[0])

        self.save_test_conds()
        results = self.initialize_results()

        for i_episode in range(self.env.config.max_episodes_test_play):
            dones = {}
            dones['all_dones'] = False
            episode_reward = 0
            episode_team_return = 0

            if self.env.config.make_time_plot:
                self.save_initial_conds()
                self.initialize_time_plot()

            while not dones['all_dones']:

                acts, scores = \
                    self.policy.sample_actions([self.padded_obss, self.padded_poss],
                                               self.mask, self.attention_mask, training=False)
                # acts:(1,n), [score1, score2]:[(1,num_heads,n,n),(1,num_heads,n,n)]

                # get alive_agents & all agents actions. action=0 <- do nothing
                actions = {}  # For alive agents

                for idx in self.alive_agents_ids:
                    agent_id = 'red_' + str(idx)
                    actions[agent_id] = acts[0, idx]

                # One step of Lanchester simulation, for alive agents in env
                # The tester does not use global state.
                next_obserations, rewards, dones, infos, reward, done, _ = self.env.step(actions)

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

                # The tester does not use global state.

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

                # agents_rewards and agents_dones, including dead and dummy ones, for result output
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

                # Update episode rewards
                episode_reward += np.sum(agents_rewards)

                episode_team_return += reward

                # Store time history of an engagement
                if self.env.config.make_time_plot:
                    self.store_time_history()

                # Make animation
                if self.env.config.make_animation:
                    self.env.make_animation.add_frame(self.env)  # log-normalized map
                    red = self.env.reds[-1]
                    if red.alive:
                        self.env.make_animation_attention_map.add_att_map(
                            relation_kernel=0,  # For relation_kernel 0
                            agent_id=red.id,
                            alive_agents_ids=self.alive_agents_ids,
                            atts=scores,
                        )

                        self.env.make_animation_attention_map.add_att_map(
                            relation_kernel=1,  # For relation_kernel 1
                            agent_id=red.id,
                            alive_agents_ids=self.alive_agents_ids,
                            atts=scores,
                        )

                        self.env.make_animation_attention_map.add_frame(
                            self.env, red.pos
                        )

                if dones['all_dones']:
                    results['episode_lens'].append(self.step)
                    results['episode_rewards'].append(episode_reward)

                    results['episode_team_return'].append(episode_team_return)

                    # Summarize each agent result
                    result_red = summarize_agent_result(self.env.reds)
                    result_blue = summarize_agent_result(self.env.blues)

                    # Decide winner
                    winner = who_wins(result_red, result_blue)

                    # Summarize episode result
                    results = summarize_episode_results(results, result_red, result_blue, winner)

                    # Generate time plot of an engagement
                    if self.env.config.make_time_plot:
                        self.make_time_plot()

                    # Generate animation
                    if self.env.config.make_animation:
                        self.env.make_animation.generate_movies(self.env)
                        self.env.make_animation_attention_map.generate_movies()

                    # Reset env
                    observations, global_state = self.env.reset()
                    self.reset_states(observations, global_state)
                else:  # dones['all_done'] ではない時
                    self.alive_agents_ids = next_alive_agents_ids
                    self.padded_obss = next_padded_obss  # (1,5,5,5,16)
                    self.padded_poss = next_padded_poss  # (1,15,8)
                    self.mask = next_mask  # (1,15)
                    self.attention_mask = next_attention_mask  # (1,15,15)

                    self.step += 1

            if i_episode % 10 == 0:
                print(f'{i_episode} episodes completed!')

        result = summarize_results(results)

        if result['num_red_win'] >= self.num_max_win:
            save_dir = Path(__file__).parent / 'models'

            save_name = '/best_win_model/'
            self.policy.save_weights(str(save_dir) + save_name)

            save_name = '/best_win_alpha'
            logalpha = current_weights[1].numpy()
            np.save(str(save_dir) + save_name, logalpha)

            self.num_max_win = result['num_red_win']

        if result['episode_rewards'] >= self.max_return:
            save_dir = Path(__file__).parent / 'models'

            save_name = '/best_return_model/'
            self.policy.save_weights(str(save_dir) + save_name)

            save_name = '/best_return_alpha'
            logalpha = current_weights[1].numpy()
            np.save(str(save_dir) + save_name, logalpha)

            self.max_return = result['episode_rewards']

        return result

    def save_initial_conds(self):
        red_properties = []
        for red in self.env.reds:
            red_properties.append({k: v for k, v in red.__dict__.items()})

        blue_properties = []
        for blue in self.env.blues:
            blue_properties.append({k: v for k, v in blue.__dict__.items()})

        initial_conds = {
            'summary of reds': {
                'R0': self.env.config.R0,
                'num_red_agents': self.env.config.num_red_agents,
                'num_red_platoons': self.env.config.num_red_platoons,
                'num_red_companies': self.env.config.num_red_companies,
            },

            'summary of blues': {
                'B0': self.env.config.B0,
                'num_blue_agents': self.env.config.num_blue_agents,
                'num_blue_platoons': self.env.config.num_blue_platoons,
                'num_blue_companies': self.env.config.num_blue_companies,
            },

            'reds_initial_properties': red_properties,
            'blues_initial_properties': blue_properties,
        }

        dir_save = './test_engagement'
        if not os.path.exists(dir_save):
            os.mkdir(dir_save)

        with open(dir_save + '/initial_conds.json', 'w') as f:
            json.dump(initial_conds, f, indent=5)

    def make_time_plot(self):
        dir_save = './test_engagement'
        if not os.path.exists(dir_save):
            os.mkdir(dir_save)

        steps = self.steps_list
        eps = 1e-3

        """ 1. platoons """
        red_platoons_num = np.array(self.red_platoons_num_list)
        red_platoons_force = np.array(self.red_platoons_force_list)
        red_platoons_efficiency = np.array(self.red_platoons_efficiency_list)
        red_platoons_ef = np.array(self.red_platoons_ef_list)

        blue_platoons_num = np.array(self.blue_platoons_num_list)
        blue_platoons_force = np.array(self.blue_platoons_force_list)
        blue_platoons_efficiency = np.array(self.blue_platoons_efficiency_list)
        blue_platoons_ef = np.array(self.blue_platoons_ef_list)

        num_red_groups = np.array(self.num_red_groups_list)

        fig1, axe1 = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(14, 8))

        axe1[0, 0].plot(steps, red_platoons_num, 'r')
        axe1[0, 0].plot(steps, blue_platoons_num, 'b')
        axe1[0, 0].set_title('Num of alive platoons')
        axe1[0, 0].grid()

        axe1[0, 1].plot(steps, red_platoons_force, 'r')
        axe1[0, 1].plot(steps, blue_platoons_force, 'b')
        axe1[0, 1].set_title('Remaining effective force of platoons')
        axe1[0, 1].grid()

        axe1[1, 0].plot(steps, red_platoons_efficiency / (red_platoons_num + eps), 'r')
        axe1[1, 0].plot(steps, blue_platoons_efficiency / (blue_platoons_num + eps), 'b')
        axe1[1, 0].set_title('Average remaining efficiency of platoons')
        axe1[1, 0].grid()

        axe1[1, 1].plot(steps, red_platoons_ef, 'r')
        axe1[1, 1].plot(steps, blue_platoons_ef, 'b')
        axe1[1, 1].set_title('Remaining efficiency * force of platoons')
        axe1[1, 1].grid()

        fig1.savefig(dir_save + '/platoons', dpi=300)

        """ 2. companies """
        red_companies_num = np.array(self.red_companies_num_list)
        red_companies_force = np.array(self.red_companies_force_list)
        red_companies_efficiency = np.array(self.red_companies_efficiency_list)
        red_companies_ef = np.array(self.red_companies_ef_list)

        blue_companies_num = np.array(self.blue_companies_num_list)
        blue_companies_force = np.array(self.blue_companies_force_list)
        blue_companies_efficiency = np.array(self.blue_companies_efficiency_list)
        blue_companies_ef = np.array(self.blue_companies_ef_list)

        fig2, axe2 = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(14, 8))

        axe2[0, 0].plot(steps, red_companies_num, 'r')
        axe2[0, 0].plot(steps, blue_companies_num, 'b')
        axe2[0, 0].set_title('Num of alive companies')
        axe2[0, 0].grid()

        axe2[0, 1].plot(steps, red_companies_force, 'r')
        axe2[0, 1].plot(steps, blue_companies_force, 'b')
        axe2[0, 1].set_title('Remaining effective force of companies')
        axe2[0, 1].grid()

        axe2[1, 0].plot(steps, red_companies_efficiency / (red_companies_num + eps), 'r')
        axe2[1, 0].plot(steps, blue_companies_efficiency / (blue_companies_num + eps), 'b')
        axe2[1, 0].set_title('Average remaining efficiency of companies')
        axe2[1, 0].grid()

        axe2[1, 1].plot(steps, red_companies_ef, 'r')
        axe2[1, 1].plot(steps, blue_companies_ef, 'b')
        axe2[1, 1].set_title('Remaining efficiency * force of companies')
        axe2[1, 1].grid()

        fig2.savefig(dir_save + '/companies', dpi=300)

        """ 3. red platoons + companies """
        fig3, axe3 = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(14, 8))

        axe3[0, 0].plot(steps, red_platoons_num + red_companies_num, 'r')
        axe3[0, 0].plot(steps, blue_platoons_num + blue_companies_num, 'b')
        axe3[0, 0].set_title('Num of alive platoons + companies')
        axe3[0, 0].grid()

        axe3[0, 1].plot(steps, red_platoons_force + red_companies_force, 'r')
        axe3[0, 1].plot(steps, blue_platoons_force + blue_companies_force, 'b')
        axe3[0, 1].set_title('Remaining effective force of platoons + companies')
        axe3[0, 1].grid()

        axe3[1, 0].plot(steps, num_red_groups, 'r')
        axe3[1, 0].set_title('Num of red clusters')
        axe3[1, 0].grid()

        axe3[1, 1].plot(steps, (red_platoons_num + red_companies_num) / num_red_groups, 'r')
        axe3[1, 1].set_title(' Num of alive red agents / Num of red clusters')
        axe3[1, 1].grid()

        fig3.savefig(dir_save + '/teams', dpi=300)

    def store_time_history(self):

        red_platoons_force = 0
        red_platoons_efficiency = 0
        red_platoons_ef = 0
        red_platoons_num = 0
        red_companies_force = 0
        red_companies_efficiency = 0
        red_companies_ef = 0
        red_companies_num = 0

        blue_platoons_force = 0
        blue_platoons_efficiency = 0
        blue_platoons_ef = 0
        blue_platoons_num = 0
        blue_companies_force = 0
        blue_companies_efficiency = 0
        blue_companies_ef = 0
        blue_companies_num = 0

        self.steps_list.append(self.step)

        for red in self.env.reds:
            if red.alive:
                if red.type == 'platoon':
                    red_platoons_force += red.effective_force
                    red_platoons_efficiency += red.efficiency
                    red_platoons_ef += red.force * red.efficiency
                    red_platoons_num += 1
                else:
                    red_companies_force += red.effective_force
                    red_companies_efficiency += red.efficiency
                    red_companies_ef += red.force * red.efficiency
                    red_companies_num += 1

        self.red_platoons_force_list.append(red_platoons_force)
        self.red_platoons_efficiency_list.append(red_platoons_efficiency)
        self.red_platoons_ef_list.append(red_platoons_ef)
        self.red_platoons_num_list.append(red_platoons_num)
        self.red_companies_force_list.append(red_companies_force)
        self.red_companies_efficiency_list.append(red_companies_efficiency)
        self.red_companies_ef_list.append(red_companies_ef)
        self.red_companies_num_list.append(red_companies_num)

        for blue in self.env.blues:
            if blue.alive:
                if blue.type == 'platoon':
                    blue_platoons_force += blue.effective_force
                    blue_platoons_efficiency += blue.efficiency
                    blue_platoons_ef += blue.force * blue.efficiency
                    blue_platoons_num += 1
                else:
                    blue_companies_force += blue.effective_force
                    blue_companies_efficiency += blue.efficiency
                    blue_companies_ef += blue.force * blue.efficiency
                    blue_companies_num += 1

        self.blue_platoons_force_list.append(blue_platoons_force)
        self.blue_platoons_efficiency_list.append(blue_platoons_efficiency)
        self.blue_platoons_ef_list.append(blue_platoons_ef)
        self.blue_platoons_num_list.append(blue_platoons_num)
        self.blue_companies_force_list.append(blue_companies_force)
        self.blue_companies_efficiency_list.append(blue_companies_efficiency)
        self.blue_companies_ef_list.append(blue_companies_ef)
        self.blue_companies_num_list.append(blue_companies_num)

        num_group = self.count_group()
        self.num_red_groups_list.append(num_group)

    def initialize_time_plot(self):
        self.steps_list = []
        self.red_platoons_force_list = []
        self.red_platoons_efficiency_list = []
        self.red_platoons_ef_list = []
        self.red_platoons_num_list = []
        self.red_companies_force_list = []
        self.red_companies_efficiency_list = []
        self.red_companies_ef_list = []
        self.red_companies_num_list = []
        self.blue_platoons_force_list = []
        self.blue_platoons_efficiency_list = []
        self.blue_platoons_ef_list = []
        self.blue_platoons_num_list = []
        self.blue_companies_force_list = []
        self.blue_companies_efficiency_list = []
        self.blue_companies_ef_list = []
        self.blue_companies_num_list = []
        self.num_red_groups_list = []

    def save_test_conds(self):
        test_conds = {
            'grid_size': self.env.config.grid_size,
            'observation_channels':self.env.config.observation_channels,
            'n_frames': self.env.config.n_frames,
            'fov': self.env.config.fov,
            'com': self.env.config.com,
            'global_grid_size': self.env.config.global_grid_size,
            'global_observation_channels': self.env.config.global_observation_channels,
            'global_n_frames': self.env.config.global_n_frames,
            'max_episodes_test_play': self.env.config.max_episodes_test_play,
            'max_steps': self.env.config.max_steps,
            'num red-platoons range': self.env.config.red_platoons,
            'num red-companies range': self.env.config.red_companies,
            'num blue-platoons range': self.env.config.blue_platoons,
            'num blue-companies range': self.env.config.blue_companies,
            'efficiency red range': self.env.config.efficiencies_red,
            'efficiency blue range': self.env.config.efficiencies_blue,
            'max num red agents': self.env.config.max_num_red_agents,
        }

        dir_save = './test_engagement'
        if not os.path.exists(dir_save):
            os.mkdir(dir_save)

        with open(dir_save + '/test_conds.json', 'w') as f:
            json.dump(test_conds, f, indent=5)

    def count_group(self):
        """ reds の cluster数をカウント"""
        x = []
        y = []
        for red in self.env.reds:
            if red.alive:
                x.append(red.pos[0])
                y.append(red.pos[1])

        x = np.array(x)
        y = np.array(y)

        group_count = 0
        for i in range(self.env.config.grid_size):
            for j in range(self.env.config.grid_size):
                num = np.sum(np.where(x == i, 1, 0) * np.where(y == j, 1, 0))
                if num > 0:
                    group_count += 1

        return group_count


def main():
    """
    Use this to make an animation.  Specify the model
    Be careful params in config, e.g.,  max_episodes_test_play=1,
                                        max_steps, n_frames, key_dim, ...
    """
    from pathlib import Path

    from config_dec_pomdp import Config

    is_debug = False  # True for debug

    if is_debug:
        print("Debug mode starts. May cause ray memory error.")
    else:
        print("Execution mode starts")

    ray.init(local_mode=is_debug, ignore_reinit_error=True)

    env = BattleFieldStrategy()
    env.reset()

    config = Config()

    grid_size = config.grid_size
    fov = config.fov
    com = config.com

    """ global_state & feature """
    global_ch = config.global_observation_channels  # 6
    global_n_frames = config.global_n_frames

    global_state_shape = \
        (config.global_grid_size, config.global_grid_size, global_ch * global_n_frames)  # (15,15,6)

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

    # Make dummy policy and load learned weights
    dummy_policy = MarlTransformerGlobalStateModel(config=config)

    dummy_policy([[padded_obs, padded_pos], global_state], mask, attention_mask, training=False)

    #""" Use the followings for the test
    # Load model
    load_dir = Path(__file__).parent / 'trial_continual/models'
    load_name = '/model_600000/'
    # load_name = '/best_return_model/'
    dummy_policy.load_weights(str(load_dir) + load_name)

    load_name = '/alpha_600000.npy'
    # load_name = '/best_return_alpha.npy'
    logalpha = np.load(str(load_dir) + load_name)
    logalpha = tf.Variable(logalpha)
    #"""
    #logalpha = tf.Variable(1.0)  # Remove this for the test

    weights = [dummy_policy.get_weights(), logalpha]

    # testerをインスタンス化
    tester = Tester.remote()

    # Start test process
    wip_tester = tester.test_play.remote(current_weights=weights)

    # Get results
    finished_tester, _ = ray.wait([wip_tester], num_returns=1)

    result = ray.get(finished_tester[0])

    print(f'{config.max_episodes_test_play} test trials:')
    print(f" - mean_episode_rewards = {result['episode_rewards']}")
    print(f" - mean_episode_len = {result['episode_lens']}")

    print(f" - mean_num_alive_reds_ratio = {result['num_alive_reds_ratio']}")
    print(f" - mean_num_alive_red_platoon = {result['num_alive_red_platoon']}"
          f" over mean_num_initial_red_platoon = {result['num_initial_red_platoon']}")
    print(f" - mean_num_alive_red_company = {result['num_alive_red_company']}"
          f" over mean_num_initial_red_company = {result['num_initial_red_company']}")
    print(f" - mean_remaining_red_effective_force_ratio = "
          f"{result['remaining_red_effective_force_ratio']}")

    print(f" - mean_num_alive_blues_ratio = {result['num_alive_blues_ratio']}")
    print(f" - mean_num_alive_blue_platoon = {result['num_alive_blue_platoon']}"
          f" over mean_num_initial_blue_platoon = {result['num_initial_blue_platoon']}")
    print(f" - mean_num_alive_blue_company = {result['num_alive_blue_company']}"
          f" over mean_num_initial_blue_company = {result['num_initial_blue_company']}")
    print(f" - mean_remaining_blue_effective_force_ratio = "
          f"{result['remaining_blue_effective_force_ratio']}")

    print(f" - num_red_win = {result['num_red_win']}")
    print(f" - num_blue_win = {result['num_blue_win']}")
    print(f" - num_draw = {result['draw']}")
    print(f" - num_no_contest = {result['no_contest']}")

    dir_save = './test_engagement'
    if not os.path.exists(dir_save):
        os.mkdir(dir_save)

    with open(dir_save + '/result.json', 'w') as f:
        json.dump(result, f, indent=5)

    ray.shutdown()


if __name__ == '__main__':
    main()
