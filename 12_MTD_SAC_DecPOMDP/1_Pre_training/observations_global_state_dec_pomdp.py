import numpy as np

from utils import add_channel_dim
from utils import compute_blue_observation_maps, compute_engage_observation_maps, \
    compute_ally_observation_maps, compute_my_observation_maps

from utils import compute_blue_observation_maps_2, compute_ally_observation_maps_2, \
    compute_my_observation_maps_2

from utils import compute_blue_observation_maps_3, compute_engage_observation_maps_3, \
    compute_ally_observation_maps_3, compute_my_observation_maps_3, compute_red_observation_maps_3

"""
    Copied from QMIX-2 (QMIX Another Implementation)
"""

""" 
global_observation = np.concatenate([
        red_normalized_force,
        red_efficiency,
        blue_normalized_force,
        blue_efficiency,
        normalized_sin(agent_id),
        normalized_cos(agent_id),
        ]), ndarray,  (grid, grid, 6), 6 CHs
"""


def get_global_observation(env):
    red_normalized_force, red_efficiency = compute_red_observation_maps_3(env)  # (g,g)
    blue_normalized_force, blue_efficiency = compute_blue_observation_maps_3(env)  # (g,g)
    sin_normalized, cos_normalized = compute_red_pos_maps(env)  # (g,g)

    red_normalized_force = add_channel_dim(red_normalized_force)  # (g,g,1)
    red_efficiency = add_channel_dim(red_efficiency)  # (g,g,1)

    blue_normalized_force = add_channel_dim(blue_normalized_force)  # (g,g,1)
    blue_efficiency = add_channel_dim(blue_efficiency)  # (g,g,1)

    sin_normalized = add_channel_dim(sin_normalized)  # (g,g,1)
    cos_normalized = add_channel_dim(cos_normalized)  # (g,g,1)

    observation = np.concatenate([
        red_normalized_force, red_efficiency,
        blue_normalized_force, blue_efficiency,
        sin_normalized, cos_normalized
    ], axis=-1)  # (g,g,6)

    return observation  # (g,g,6)


def compute_red_pos_maps(env):
    """
    map of agent position in agent set
    """
    sin_map = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)
    cos_map = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)

    i = 0

    for red in env.reds:
        if red.alive:
            sin_map[red.pos[0], red.pos[1]] += (np.sin(i / 10.) + 1.) / 2.
            cos_map[red.pos[0], red.pos[1]] += (np.cos(i / 10.) + 1.) / 2.

            i += 1

    normalized_sin_map = 2.0 / np.pi * np.arctan(sin_map)
    normalized_cos_map = 2.0 / np.pi * np.arctan(cos_map)

    return normalized_sin_map, normalized_cos_map


def get_observations(env):
    """ Select one of the followings """

    observations = get_observations_po_0(env)  # 6 channels, new observation scale

    return observations


"""
agent egocentric observations = {agent_id: obs}, dict

obs = [
    ally_normalized_force
    ally_efficiency
    blue_normalized_force
    blue_efficiency
    ], list
"""


def get_observations_po_0(env):
    """
    observations: {agent-id: (env.config.fov * 2 + 1,
                              env.config.fov * 2 + 1,
                              env.config.observation_channels)}
     normalized by arc-tan().  4CH
    """

    if env.config.observation_channels != 4:
        raise ValueError()

    fov = env.config.fov
    observations = {}

    for red in env.reds:
        if red.alive:
            myx = red.pos[0]
            myy = red.pos[1]

            red_normalized_force, red_efficiency = \
                compute_partial_observation_map(myx, myy, env.reds, fov)

            blue_normalized_force, blue_efficiency = \
                compute_partial_observation_map(myx, myy, env.blues, fov)

            # transform to float32 & add channel dim
            red_normalized_force = add_channel_dim(red_normalized_force)
            red_efficiency = add_channel_dim(red_efficiency)

            blue_normalized_force = add_channel_dim(blue_normalized_force)
            blue_efficiency = add_channel_dim(blue_efficiency)

            observations[red.id] = np.concatenate(
                [
                    red_normalized_force,
                    red_efficiency,
                    blue_normalized_force,
                    blue_efficiency,
                ], axis=2)  # (5,5,4)

    return observations


def compute_partial_observation_map(myx, myy, agents, fov):
    """
    (myx, myy) : myself global pos
    agents: reds or blues
    fov: field of view

    :return: normalized_force_map, efficiency_map
    """

    force_map = np.zeros((2 * fov + 1, 2 * fov + 1))
    ef_map = np.zeros((2 * fov + 1, 2 * fov + 1))

    for x in range(myx - fov, myx + fov + 1):
        for y in range(myy - fov, myy + fov + 1):
            for agent in agents:
                if agent.alive and (agent.pos[0] == x) and (agent.pos[1] == y):
                    force_map[agent.pos[0] - myx + fov, agent.pos[1] - myy + fov] += agent.force
                    ef_map[agent.pos[0] - myx + fov, agent.pos[1] - myy + fov] += agent.ef

    # normalize
    alpha = 50.
    normalized_force_map = 2.0 / np.pi * np.arctan(force_map / alpha)

    efficiency_map = np.zeros((2 * fov + 1, 2 * fov + 1))

    xf, yf = np.nonzero(force_map)
    for (x, y) in zip(xf, yf):
        efficiency_map[x, y] = ef_map[x, y] / force_map[x, y]

    return normalized_force_map, efficiency_map
