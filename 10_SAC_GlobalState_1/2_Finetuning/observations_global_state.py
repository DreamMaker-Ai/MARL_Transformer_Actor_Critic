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


"""
agent observations = {agent_id: obs}, dict

obs = [
    ally_normalized_force
    ally_efficiency
    my_normalized_force
    my_efficiency
    blue_normalized_force
    blue_efficiency
    ], list
"""


def get_observations(env):
    """ Select one of the followings """

    observations = get_observations_0(env)  # 6 channels, new observation scale

    # observations = get_observations_1(env)  # 8 channels

    # observations = get_observations_2(env)  # 11 channels

    # observations = get_observations_3(env)  # 8 channels, new observation scale

    return observations


def get_observations_0(env):
    """
    observations: {agent-id: (grid_size, grid_size, env.config.observation_channels)}
     normalized by arc-tan().  6CH
    """

    if env.config.observation_channels != 6:
        raise ValueError()

    observations = {}

    """ blue channels, all red agents share the same map """
    blue_normalized_force, blue_efficiency = compute_blue_observation_maps_3(env)

    """ engagement channels, all red agents share the same map """
    # engage_normalized_force = compute_engage_observation_maps_3(env)

    # transform to float32 & add channel dim
    # battlefield = add_channel_dim(env.battlefield)

    blue_normalized_force = add_channel_dim(blue_normalized_force)
    blue_efficiency = add_channel_dim(blue_efficiency)

    # engage_normalized_force = add_channel_dim(engage_normalized_force)

    """ red ally and myself, each red agent has the different map """

    for red in env.reds:
        if red.alive:
            """ ally """
            ally_normalized_force, ally_efficiency = compute_ally_observation_maps_3(red, env)

            """ myself """
            my_normalized_force, my_efficiency = compute_my_observation_maps_3(red, env)

            # transform to float32 & add channel dim
            ally_normalized_force = add_channel_dim(ally_normalized_force)
            ally_efficiency = add_channel_dim(ally_efficiency)

            my_normalized_force = add_channel_dim(my_normalized_force)
            my_efficiency = add_channel_dim(my_efficiency)

            observations[red.id] = np.concatenate(
                [
                    ally_normalized_force,
                    ally_efficiency,
                    my_normalized_force,
                    my_efficiency,
                    blue_normalized_force,
                    blue_efficiency,
                    # engage_normalized_force,
                ], axis=2)

    return observations


def get_observations_1(env):
    """
    observations: {agent-id: (grid_size, grid_size, env.config.observation_channels)}
     normalized by initial ef and force.
    """

    if env.config.observation_channels != 8:
        raise ValueError()

    observations = {}

    """ blue channels, all red agents share the same map """
    blue_log_normalized_force, blue_efficiency = compute_blue_observation_maps(env)

    """ engagement channels, all red agents share the same map """
    engage_log_normalized_force = compute_engage_observation_maps(env)

    # transform to float32 & add channel dim
    battlefield = add_channel_dim(env.battlefield)

    blue_log_normalized_force = add_channel_dim(blue_log_normalized_force)
    blue_efficiency = add_channel_dim(blue_efficiency)

    engage_log_normalized_force = add_channel_dim(engage_log_normalized_force)

    """ red ally and myself, each red agent has the different map """

    for red in env.reds:
        if red.alive:
            """ ally """
            ally_log_normalized_force, ally_efficiency = compute_ally_observation_maps(red, env)

            """ myself """
            my_log_normalized_force, my_efficiency = compute_my_observation_maps(red, env)

            # transform to float32 & add channel dim
            ally_log_normalized_force = add_channel_dim(ally_log_normalized_force)
            ally_efficiency = add_channel_dim(ally_efficiency)

            my_log_normalized_force = add_channel_dim(my_log_normalized_force)
            my_efficiency = add_channel_dim(my_efficiency)

            observations[red.id] = np.concatenate(
                [battlefield,
                 ally_log_normalized_force,
                 ally_efficiency,
                 my_log_normalized_force,
                 my_efficiency,
                 blue_log_normalized_force,
                 blue_efficiency,
                 engage_log_normalized_force,
                 ], axis=2)

    return observations


def get_observations_2(env):
    """
    observations: {agent-id: (grid_size, grid_size, env.config.observation_channels)}
     normalized by initial ef and force.
    Ally, myself, blue position maps are added
    """

    if env.config.observation_channels != 11:
        raise ValueError()

    observations = {}

    """ blue channels, all red agents share the same map """
    blue_log_normalized_force, blue_efficiency, blue_pos = compute_blue_observation_maps_2(env)

    """ engagement channels, all red agents share the same map """
    engage_log_normalized_force = compute_engage_observation_maps(env)

    # transform to float32 & add channel dim
    battlefield = add_channel_dim(env.battlefield)

    blue_log_normalized_force = add_channel_dim(blue_log_normalized_force)
    blue_efficiency = add_channel_dim(blue_efficiency)
    blue_pos = add_channel_dim(blue_pos)

    engage_log_normalized_force = add_channel_dim(engage_log_normalized_force)

    """ red ally and myself, each red agent has the different map """

    for red in env.reds:
        if red.alive:
            """ ally """
            ally_log_normalized_force, ally_efficiency, ally_pos = \
                compute_ally_observation_maps_2(red, env)

            """ myself """
            my_log_normalized_force, my_efficiency, my_pos = \
                compute_my_observation_maps_2(red, env)

            # transform to float32 & add channel dim
            ally_log_normalized_force = add_channel_dim(ally_log_normalized_force)
            ally_efficiency = add_channel_dim(ally_efficiency)
            ally_pos = add_channel_dim(ally_pos)

            my_log_normalized_force = add_channel_dim(my_log_normalized_force)
            my_efficiency = add_channel_dim(my_efficiency)
            my_pos = add_channel_dim(my_pos)

            observations[red.id] = np.concatenate(
                [battlefield,
                 ally_log_normalized_force,
                 ally_efficiency,
                 my_log_normalized_force,
                 my_efficiency,
                 blue_log_normalized_force,
                 blue_efficiency,
                 engage_log_normalized_force,
                 ally_pos,
                 my_pos,
                 blue_pos
                 ], axis=2)  # 11 channels

    return observations


def get_observations_3(env):
    """
    observations: {agent-id: (grid_size, grid_size, env.config.observation_channels)}
     normalized by arc-tan().  8CH
    """

    if env.config.observation_channels != 8:
        raise ValueError()

    observations = {}

    """ blue channels, all red agents share the same map """
    blue_normalized_force, blue_efficiency = compute_blue_observation_maps_3(env)

    """ engagement channels, all red agents share the same map """
    engage_normalized_force = compute_engage_observation_maps_3(env)

    # transform to float32 & add channel dim
    battlefield = add_channel_dim(env.battlefield)

    blue_normalized_force = add_channel_dim(blue_normalized_force)
    blue_efficiency = add_channel_dim(blue_efficiency)

    engage_normalized_force = add_channel_dim(engage_normalized_force)

    """ red ally and myself, each red agent has the different map """

    for red in env.reds:
        if red.alive:
            """ ally """
            ally_normalized_force, ally_efficiency = compute_ally_observation_maps_3(red, env)

            """ myself """
            my_normalized_force, my_efficiency = compute_my_observation_maps_3(red, env)

            # transform to float32 & add channel dim
            ally_normalized_force = add_channel_dim(ally_normalized_force)
            ally_efficiency = add_channel_dim(ally_efficiency)

            my_normalized_force = add_channel_dim(my_normalized_force)
            my_efficiency = add_channel_dim(my_efficiency)

            observations[red.id] = np.concatenate(
                [battlefield,
                 ally_normalized_force,
                 ally_efficiency,
                 my_normalized_force,
                 my_efficiency,
                 blue_normalized_force,
                 blue_efficiency,
                 engage_normalized_force,
                 ], axis=2)

    return observations
