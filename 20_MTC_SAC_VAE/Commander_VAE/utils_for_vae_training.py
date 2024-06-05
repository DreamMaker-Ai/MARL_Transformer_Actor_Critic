import numpy as np
import math


def compute_global_blue_maps(env):
    """
    Compute blues log normalized force and efficiency 2d maps for global state.
    :return:
        blue_normalized_force
        blue_efficiency
    """

    blue_ef = np.zeros(
        (env.config.global_grid_size, env.config.global_grid_size), dtype=np.float32)
    blue_force = np.zeros(
        (env.config.global_grid_size, env.config.global_grid_size), dtype=np.float32)
    blue_effective_force = np.zeros(
        (env.config.global_grid_size, env.config.global_grid_size), dtype=np.float32)

    ratio = env.config.global_grid_size / env.config.grid_size

    for blue in env.blues:
        if blue.alive:
            x = math.floor(blue.pos[0] * ratio)
            y = math.floor(blue.pos[1] * ratio)
            blue_ef[x, y] += blue.ef
            blue_force[x, y] += blue.force
            blue_effective_force[x, y] += blue.effective_force

    xf, yf = np.nonzero(blue_force)

    # compute normalized_force  only for alive blue
    alpha = 50.0
    blue_normalized_force = 2.0 / np.pi * np.arctan(blue_force / alpha)

    # compute efficiency
    blue_efficiency = \
        compute_global_cell_efficiency(xf, yf, blue_force, blue_ef,
                                       env.config.grid_size, env.config.global_grid_size)

    return blue_normalized_force, blue_efficiency


def compute_global_red_maps(env):
    """
    Compute reds log normalized force and efficiency 2d maps for global state.
    :return:
        red_normalized_force
        red_efficiency
    """

    red_ef = np.zeros(
        (env.config.global_grid_size, env.config.global_grid_size), dtype=np.float32)
    red_force = np.zeros(
        (env.config.global_grid_size, env.config.global_grid_size), dtype=np.float32)
    red_effective_force = np.zeros(
        (env.config.global_grid_size, env.config.global_grid_size), dtype=np.float32)

    ratio = env.config.global_grid_size / env.config.grid_size

    for red in env.reds:
        if red.alive:
            x = math.floor(red.pos[0] * ratio)
            y = math.floor(red.pos[1] * ratio)

            red_ef[x, y] += red.ef
            red_force[x, y] += red.force
            red_effective_force[x, y] += red.effective_force

    xf, yf = np.nonzero(red_force)

    # compute normalized_force  only for alive red
    alpha = 50.0
    red_normalized_force = 2.0 / np.pi * np.arctan(red_force / alpha)

    # compute efficiency
    red_efficiency = \
        compute_global_cell_efficiency(xf, yf, red_force, red_ef,
                                       env.config.grid_size, env.config.global_grid_size)

    return red_normalized_force, red_efficiency


def compute_global_cell_efficiency(xf, yf, force, ef, grid_size, global_grid_size):
    # For observation
    efficiency = np.zeros((global_grid_size, global_grid_size), dtype=np.float32)

    ratio = global_grid_size / grid_size

    for (x, y) in zip(xf, yf):
        gx = math.floor(x * ratio)
        gy = math.floor(y * ratio)
        efficiency[gx, gy] = ef[x, y] / force[x, y]

    # check
    if (np.any(efficiency) < 0) or (np.any(efficiency) > 1):
        raise ValueError()

    return efficiency


def add_channel_dim(map_2D):
    return np.expand_dims(map_2D, axis=2).astype(np.float32)
