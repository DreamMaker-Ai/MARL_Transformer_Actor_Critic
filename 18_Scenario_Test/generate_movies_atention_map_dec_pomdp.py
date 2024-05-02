import copy

import numpy as np
import re
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

from utils import add_channel_dim
from utils import compute_current_total_ef_and_force, add_channel_dim, \
    count_alive_agents, count_alive_platoons_and_companies
from utils import compute_red_observation_maps_3, compute_blue_observation_maps_3, \
    compute_engage_observation_maps_3


def rgb_channel_maker(r_channel, g_channel, b_channel):
    """
    Make RGB numpy array: (grid_size,grid_size, 3)
    """
    rgb_channel = np.concatenate([r_channel, g_channel, b_channel], axis=2)

    if np.max(rgb_channel) > 1 or np.min(rgb_channel) < 0:
        raise ValueError

    return rgb_channel


class MakeAnimation_AttentionMap:
    def __init__(self, env):
        self.env = env

        battlefield = env.battlefield  # (grid,grid)=(20,20)
        self.battlefield = add_channel_dim(battlefield)

        self.num_red_agents = self.env.config.num_red_agents
        self.dt = self.env.config.dt

        self.rgb_channel_forces = []  # observation map of effective forces
        self.rgb_channel_efficiencies = []  # observation map of efficiencies
        self.rgb_channel_engage_forces = []  # observation map of engage forces

        self.attention_maps = [[], []]
        self.num_heads = self.env.config.num_heads  # 2

        self.num_alive_reds = []  # number of alive red
        self.num_alive_blues = []

        self.num_alive_reds_platoons = []
        self.num_alive_reds_companies = []

        self.num_alive_blues_platoons = []
        self.num_alive_blues_companies = []

        self.total_efficiency_reds = []  # efficiency r
        self.total_efficiency_blues = []  # effciency b

        self.total_force_reds = []  # effective R
        self.total_force_blues = []  # effective B

    def add_att_map(self, relation_kernel, agent_id, alive_agents_ids, atts):
        # atts: list of the attention_matrix of relation_kernels
        #
        #

        g_size = self.battlefield.shape[0]  # 20

        i = int(re.sub(r"\D", "", agent_id))  # index of agent_i, int

        att_scores = atts[relation_kernel]  # (1,num_heads,n,n), n=max_num_red_agents

        att_score = att_scores[0, :, i, :]  # Attention matrix of agent_i, (2,n)

        attention_map = np.zeros([g_size, g_size, 3])  # RGB-array

        for head in range(self.num_heads):
            att_vals = att_score[head, :]  # (n,)

            for a in alive_agents_ids:
                dist_x = np.abs(self.env.reds[i].pos[0] - self.env.reds[a].pos[0])
                dist_y = np.abs(self.env.reds[i].pos[1] - self.env.reds[a].pos[1])

                if dist_x <= self.env.config.com and dist_y <= self.env.config.com:
                    att_val = att_vals[a]
                    att_pos = self.env.reds[a].pos

                    attention_map[att_pos[0], att_pos[1], head % 3] += att_val

        self.attention_maps[relation_kernel].append(attention_map)

    def generate_movies(self):
        """
        heads_type: 'val_heads' or 'policy_heads'
        """
        scenario_id = str(self.env.config.scenario_id)
        dir_save = './test_scenario_' + scenario_id

        fontsize = 12
        ims = []

        num_subplots = self.num_heads

        num_colms = int(np.ceil(np.sqrt(num_subplots)))  # 2
        num_rows = int(np.ceil(num_subplots / num_colms))  # 2

        fig = plt.figure(figsize=(19.2, 9.6), tight_layout=True)
        gs = gridspec.GridSpec(num_rows, num_colms + 1)

        for step in range(len(self.attention_maps[0])):
            im = []

            # Draw attention score of each heads
            for row in range(num_rows):
                for colm in range(num_colms):
                    relation_kernel = row * num_colms + colm

                    plt.subplot(gs[row, colm])
                    plt.title(str('relation kernel-' + str(relation_kernel) +
                                  ':     head_0:red,   head_1:green'))
                    plt.tick_params(labelbottom=False, bottom=False,
                                    labelleft=False, left=False)
                    img = plt.imshow(self.attention_maps[relation_kernel][step],
                                     vmin=0, vmax=1, animated=True)

                    im += [img]

            txt1 = plt.text(0.0, -3.0, ('time:' + str(np.round(self.dt * step, 2)) + ' sec'),
                            fontsize=fontsize)

            # Draw force map
            plt.subplot(gs[:, num_colms])
            plt.title('remaining force')
            plt.tick_params(labelbottom=False, bottom=False, labelleft=False, left=False)

            img = plt.imshow(self.rgb_channel_forces[step], vmin=0, vmax=1, animated=True)

            im += [img]

            ims.append(im + [txt1])

        anim = animation.ArtistAnimation(fig, ims, interval=300, blit=True,
                                         repeat_delay=3000, repeat=True)

        scenario_id = str(self.env.config.scenario_id)
        dir_save = './test_scenario_' + scenario_id

        filename = dir_save + '/agent_obs_attention_heads'
        anim.save(filename + '.mp4', writer='ffmpeg')
        # For mp4, need 'sudo apt install ffmpeg' @ terminal

    def add_frame(self, env, agent_pos):
        """
        Call this method every time_step
        """
        # 2D map of effective ef and effective force: # (grid_size,grid_size)
        red_normalized_force, red_efficiency = compute_red_observation_maps_3(env)
        blue_normalized_force, blue_efficiency = compute_blue_observation_maps_3(env)

        engage_normalized_force = compute_engage_observation_maps_3(env)

        # Add channel dim : (grid_size,grid_size,1)
        red_normalized_force = add_channel_dim(red_normalized_force)
        red_efficiency = add_channel_dim(red_efficiency)
        blue_normalized_force = add_channel_dim(blue_normalized_force)
        blue_efficiency = add_channel_dim(blue_efficiency)

        engage_normalized_force = add_channel_dim(engage_normalized_force)

        """ normalized(force) map; (grid_size,grid_size,3) """
        r_channel_force = self.battlefield + red_normalized_force
        g_channel_force = copy.copy(self.battlefield)
        b_channel_force = self.battlefield + blue_normalized_force

        # my position (Yellow)
        r_channel_force[agent_pos[0], agent_pos[1], 0] = 1.0
        g_channel_force[agent_pos[0], agent_pos[1], 0] = 1.0
        b_channel_force[agent_pos[0], agent_pos[1], 0] = 0.0

        rgb_channel_force = rgb_channel_maker(r_channel_force, g_channel_force, b_channel_force)

        self.rgb_channel_forces.append(rgb_channel_force)

        """ efficiency map; (grid_size,grid_size,3) """
        r_channel_efficiency = self.battlefield + red_efficiency
        g_channel_efficiency = self.battlefield
        b_channel_efficiency = self.battlefield + blue_efficiency

        rgb_channel_efficiency = \
            rgb_channel_maker(r_channel_efficiency, g_channel_efficiency, b_channel_efficiency)
        self.rgb_channel_efficiencies.append(rgb_channel_efficiency)

        """ normalized(engage force) map; (grid_size,grid_size,3) """
        r_channel_engage_force = self.battlefield
        g_channel_engage_force = self.battlefield + engage_normalized_force
        b_channel_engage_force = self.battlefield

        rgb_channel_engage_force = rgb_channel_maker(
            r_channel_engage_force, g_channel_engage_force, b_channel_engage_force)

        self.rgb_channel_engage_forces.append(rgb_channel_engage_force)

        """ Add number of alive agents """
        num_alive_reds = count_alive_agents(env.reds)
        num_alive_blues = count_alive_agents(env.blues)

        self.num_alive_reds.append(num_alive_reds)
        self.num_alive_blues.append(num_alive_blues)

        num_alive_reds_platoons, num_alive_reds_companies = \
            count_alive_platoons_and_companies(env.reds)
        num_alive_blues_platoons, num_alive_blues_companies = \
            count_alive_platoons_and_companies(env.blues)

        self.num_alive_reds_platoons.append(num_alive_reds_platoons)
        self.num_alive_reds_companies.append(num_alive_reds_companies)

        self.num_alive_blues_platoons.append(num_alive_blues_platoons)
        self.num_alive_blues_companies.append(num_alive_blues_companies)

        """ Add total effective_ef and effective_force """
        # effective rR, effective R, effective bB, effective B
        (total_ef_reds, total_force_reds,
         total_effective_ef_reds, total_effective_force_reds) = \
            compute_current_total_ef_and_force(env.reds)
        (total_ef_blues, total_force_blues,
         total_effective_ef_blues, total_effective_force_blues) = \
            compute_current_total_ef_and_force(env.blues)

        self.total_efficiency_reds.append(total_ef_reds / (total_force_reds + 1e-5))
        self.total_efficiency_blues.append(total_ef_blues / (total_force_blues + 1e-5))

        self.total_force_reds.append(total_effective_force_reds)
        self.total_force_blues.append(total_effective_force_blues)
