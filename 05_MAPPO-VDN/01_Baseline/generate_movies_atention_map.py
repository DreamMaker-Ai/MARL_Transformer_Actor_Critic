import numpy as np
import re
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

from utils import add_channel_dim


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

        self.rgb_channel_forces_obs = []  # observation map of effective forces
        self.rgb_channel_efficiencies_obs = []  # observation map of efficiencies
        self.rgb_channel_engage_forces_obs = [] # observation map of engage forces

        self.attention_maps = [[], []]
        self.num_heads = self.env.config.num_heads  # 2

    def add_att_map(self, relation_kernel, agent_id, alive_agents_ids, atts):
        # atts: list of the attention_matrix of relation_kernels
        g_size = self.battlefield.shape[0]  # 20

        i = int(re.sub(r"\D", "", agent_id))  # index of agent_i, int

        att_scores = atts[relation_kernel]  # (1,num_heads,n,n), n=max_num_red_agents

        att_score = att_scores[0, :, i, :]  # Attention matrix of agent_i, (2,n)

        attention_map = np.zeros([g_size, g_size, 3])  # RGB-array

        for head in range(self.num_heads):
            att_vals = att_score[head, :]  # (n,)

            for j, a in enumerate(alive_agents_ids):
                att_val = att_vals[j]
                att_pos = self.env.reds[a].pos

                attention_map[att_pos[0], att_pos[1], head % 3] += att_val

        self.attention_maps[relation_kernel].append(attention_map)

    def add_observations(self, observations):
        """
        observations of red[-1]: (grid_size, grid_size, env.config.observation_channels)
        observations = [
            0. env.battlefield
            1. ally_normalized_force
            2. ally_efficiency
            3. my_normalized_force
            4. my_efficiency
            5. blue_normalized_force
            6. blue_efficiency
            7. engage_normalized_force ]
        """
        # observations of log-normalized force
        r_channel_force = observations[:, :, 0] + observations[:, :, 1] + observations[:, :, 3]
        g_channel_force = observations[:, :, 0] + observations[:, :, 3]
        b_channel_force = observations[:, :, 0] + observations[:, :, 5]

        r_channel_force = np.clip(r_channel_force, a_min=0, a_max=1)
        g_channel_force = np.clip(g_channel_force, a_min=0, a_max=1)

        # observations of efficiency
        r_channel_efficiency = observations[:, :, 0] + observations[:, :, 2] + observations[:, :, 4]
        g_channel_efficiency = observations[:, :, 0] + observations[:, :, 4]
        b_channel_efficiency = observations[:, :, 0] + observations[:, :, 6]

        r_channel_efficiency = np.clip(r_channel_efficiency, a_min=0, a_max=1)
        g_channel_efficiency = np.clip(g_channel_efficiency, a_min=0, a_max=1)

        # observations of log-normalized engage force
        r_channel_engage_force = observations[:, :, 0]
        g_channel_engage_force = observations[:, :, 0] + observations[:, :, 7]
        b_channel_engage_force = observations[:, :, 0]

        r_channel_engage_force = np.clip(r_channel_engage_force, a_min=0, a_max=1)

        # Add channel dim
        r_channel_force = add_channel_dim(r_channel_force)
        g_channel_force = add_channel_dim(g_channel_force)
        b_channel_force = add_channel_dim(b_channel_force)

        r_channel_efficiency = add_channel_dim(r_channel_efficiency)
        g_channel_efficiency = add_channel_dim(g_channel_efficiency)
        b_channel_efficiency = add_channel_dim(b_channel_efficiency)

        r_channel_engage_force = add_channel_dim(r_channel_engage_force)
        g_channel_engage_force = add_channel_dim(g_channel_engage_force)
        b_channel_engage_force = add_channel_dim(b_channel_engage_force)

        # Make RGB
        rgb_channel_force = rgb_channel_maker(r_channel_force, g_channel_force, b_channel_force)

        rgb_channel_efficiency = \
            rgb_channel_maker(r_channel_efficiency, g_channel_efficiency, b_channel_efficiency)

        rgb_channel_engage_force = \
            rgb_channel_maker(
                r_channel_engage_force, g_channel_engage_force, b_channel_engage_force)

        # Append
        self.rgb_channel_forces_obs.append(rgb_channel_force)
        self.rgb_channel_efficiencies_obs.append(rgb_channel_efficiency)

    def generate_movies(self):
        """
        heads_type: 'val_heads' or 'policy_heads'
        """
        dir_save = './test_engagement'
        if not os.path.exists(dir_save):
            os.mkdir(dir_save)

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

            img = plt.imshow(self.rgb_channel_forces_obs[step], vmin=0, vmax=1, animated=True)

            im += [img]

            ims.append(im + [txt1])

        anim = animation.ArtistAnimation(fig, ims, interval=300, blit=True,
                                         repeat_delay=3000, repeat=True)

        filename = './test_engagement/agent_obs_attention_heads'
        anim.save(filename + '.mp4', writer='ffmpeg')
        # For mp4, need 'sudo apt install ffmpeg' @ terminal


    def add_observations_3(self, observations):
        """
        For 6 channels observation
        observations of red[-1]: (grid_size, grid_size, env.config.observation_channels)
        observations = [
            0. ally_normalized_force
            1. ally_efficiency
            2. my_normalized_force
            3. my_efficiency
            4. blue_normalized_force
            5. blue_efficiency
        """
        # observations of normalized force
        r_channel_force = self.battlefield[:, :, 0] + observations[:, :, 0] + observations[:, :, 2]
        g_channel_force = self.battlefield[:, :, 0] + observations[:, :, 2]
        b_channel_force = self.battlefield[:, :, 0] + observations[:, :, 4]

        r_channel_force = np.clip(r_channel_force, a_min=0, a_max=1)

        # observations of efficiency
        r_channel_efficiency = self.battlefield[:, :, 0] +\
                               observations[:, :, 1] + observations[:, :, 3]
        g_channel_efficiency = self.battlefield[:, :, 0] + observations[:, :, 3]
        b_channel_efficiency = self.battlefield[:, :, 0] + observations[:, :, 5]

        r_channel_efficiency = np.clip(r_channel_efficiency, a_min=0, a_max=1)

        # observations of normalized engage force
        r_channel_engage_force = self.battlefield[:, :, 0]
        g_channel_engage_force = self.battlefield[:, :, 0] + r_channel_force * b_channel_force
        b_channel_engage_force = self.battlefield[:, :, 0]

        r_channel_engage_force = np.clip(r_channel_engage_force, a_min=0, a_max=1)

        # Add channel dim
        r_channel_force = add_channel_dim(r_channel_force)
        g_channel_force = add_channel_dim(g_channel_force)
        b_channel_force = add_channel_dim(b_channel_force)

        r_channel_efficiency = add_channel_dim(r_channel_efficiency)
        g_channel_efficiency = add_channel_dim(g_channel_efficiency)
        b_channel_efficiency = add_channel_dim(b_channel_efficiency)

        r_channel_engage_force = add_channel_dim(r_channel_engage_force)
        g_channel_engage_force = add_channel_dim(g_channel_engage_force)
        b_channel_engage_force = add_channel_dim(b_channel_engage_force)

        # Make RGB
        rgb_channel_force = rgb_channel_maker(r_channel_force, g_channel_force, b_channel_force)

        rgb_channel_efficiency = \
            rgb_channel_maker(r_channel_efficiency, g_channel_efficiency, b_channel_efficiency)

        rgb_channel_engage_force = \
            rgb_channel_maker(
                r_channel_engage_force, g_channel_engage_force, b_channel_engage_force)

        # Append
        self.rgb_channel_forces_obs.append(rgb_channel_force)
        self.rgb_channel_efficiencies_obs.append(rgb_channel_efficiency)

        self.rgb_channel_engage_forces_obs.append(rgb_channel_engage_force)
