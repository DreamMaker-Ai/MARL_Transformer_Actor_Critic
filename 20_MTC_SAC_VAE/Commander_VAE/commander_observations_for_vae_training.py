import numpy as np
import math
import cv2

from utils_for_vae_training import add_channel_dim
from utils_for_vae_training import compute_global_blue_maps, compute_global_red_maps

""" 
commander_observation = np.concatenate([
        red_normalized_force,
        red_efficiency,
        blue_normalized_force,
        blue_efficiency,
        normalized_sin(agent_id),
        normalized_cos(agent_id),
        ]), ndarray,  (grid, grid, 6), 6 CHs
"""


def get_commander_observation(env):  # For reds commander_state
    red_normalized_force, red_efficiency = compute_global_red_maps(env)  # (g,g)
    blue_normalized_force, blue_efficiency = compute_global_blue_maps(env)  # (g,g)
    sin_normalized, cos_normalized = compute_global_red_pos_maps(env)  # (g,g)

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


def get_blue_commander_observation(env):  # For reds commander_state
    red_normalized_force, red_efficiency = compute_global_red_maps(env)  # (g,g)
    blue_normalized_force, blue_efficiency = compute_global_blue_maps(env)  # (g,g)
    sin_normalized, cos_normalized = compute_global_blue_pos_maps(env)  # (g,g)

    red_normalized_force = add_channel_dim(red_normalized_force)  # (g,g,1)
    red_efficiency = add_channel_dim(red_efficiency)  # (g,g,1)

    blue_normalized_force = add_channel_dim(blue_normalized_force)  # (g,g,1)
    blue_efficiency = add_channel_dim(blue_efficiency)  # (g,g,1)

    sin_normalized = add_channel_dim(sin_normalized)  # (g,g,1)
    cos_normalized = add_channel_dim(cos_normalized)  # (g,g,1)

    observation = np.concatenate([
        blue_normalized_force, blue_efficiency,
        red_normalized_force, red_efficiency,
        sin_normalized, cos_normalized
    ], axis=-1)  # (g,g,6)

    return observation  # (g,g,6)


def commander_state_resize(state, commander_grid_size):
    resized_state = cv2.resize(state,
                               dsize=(commander_grid_size, commander_grid_size),
                               interpolation=cv2.INTER_LINEAR)

    return resized_state


def compute_global_red_pos_maps(env):  # For reds global_state
    """
    map of red_agent_id distribution
    """

    sin_map = np.zeros((env.config.global_grid_size, env.config.global_grid_size), dtype=np.float32)
    cos_map = np.zeros((env.config.global_grid_size, env.config.global_grid_size), dtype=np.float32)

    i = 0

    ratio = env.config.global_grid_size / env.config.grid_size

    for red in env.reds:
        if red.alive:
            x = math.floor(red.pos[0] * ratio)
            y = math.floor(red.pos[1] * ratio)

            sin_map[x, y] += (np.sin(i / 100.) + 1.) / 2.
            cos_map[x, y] += (np.cos(i / 100.) + 1.) / 2.

            i += 1

    normalized_sin_map = 2.0 / np.pi * np.arctan(sin_map)
    normalized_cos_map = 2.0 / np.pi * np.arctan(cos_map)

    return normalized_sin_map, normalized_cos_map


def compute_global_blue_pos_maps(env):  # For blues global_state
    """
    map of blue_agent_id distribution
    """

    sin_map = np.zeros((env.config.global_grid_size, env.config.global_grid_size), dtype=np.float32)
    cos_map = np.zeros((env.config.global_grid_size, env.config.global_grid_size), dtype=np.float32)

    i = 0

    ratio = env.config.global_grid_size / env.config.grid_size

    for blue in env.blues:
        if blue.alive:
            x = math.floor(blue.pos[0] * ratio)
            y = math.floor(blue.pos[1] * ratio)

            sin_map[x, y] += (np.sin(i / 100.) + 1.) / 2.
            cos_map[x, y] += (np.cos(i / 100.) + 1.) / 2.

            i += 1

    normalized_sin_map = 2.0 / np.pi * np.arctan(sin_map)
    normalized_cos_map = 2.0 / np.pi * np.arctan(cos_map)

    return normalized_sin_map, normalized_cos_map


