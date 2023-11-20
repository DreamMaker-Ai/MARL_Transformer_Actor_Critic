"""
Copied from
    MARL_Transformer_APEX_DQN/G_MARL_Transformer_Centralized_QMIX/02_FineTuning/
    battlefield_strategy_rev10_finetuning.py, then modified l.15-16
"""
import gym
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pprint
import pickle

from agents_in_env import RED, BLUE
from config import Config
from generate_agents_in_env import generate_red_team, generate_blue_team

from rewards import get_consolidation_of_force_rewards, get_economy_of_force_rewards
from observations import get_observations
from engage import engage_and_get_rewards, compute_engage_mask, get_dones

from generate_movies import MakeAnimation
from utils import compute_current_total_ef_and_force, count_alive_agents

from generate_movies_atention_map import MakeAnimation_AttentionMap


class BattleFieldStrategy(gym.Env):
    def __init__(self):
        super(BattleFieldStrategy, self).__init__()

        self.config = Config()

        self.action_space = self.config.action_space

        self.observation_space = self.config.observation_space

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

        self.make_animation = MakeAnimation(self)  # Animation generator

        self.make_animation_attention_map = MakeAnimation_AttentionMap(self)  # Attention map movie

        return observations

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

    def move_blues(self):
        """ TBD """
        pass

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

        # 1. move to the new cell
        self.move_reds(actions)
        self.move_blues()

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

        # 5. Get (next) observations after engagement
        observations = get_observations(self)

        return observations, rewards, dones, infos, reward, done

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
