import numpy as np
import os
import re
import pickle
import matplotlib.pyplot as plt
from collections import deque
import json
from pathlib import Path

"""
summary_of_team

# For preliminary Lanchester simulation
simulate_lanchester 
 - plot_force_history
 - plot_efficiency_history
 - visualize_blues_num_map
 - visualize_reds_num_map
 - visualize_battlefield_agents_num_map

compute_sum_of_rewards
count_alive_agents
count_alive_platoons_and_companies
compute_current_total_ef_and_force

compute_log_normalized_force

# For observation
 - compute_engage_mask
 - add_channel_dim

 - compute_log_normalized_cell_force
 - compute_cell_efficiency
 - compute_red_observation_maps
 - compute_blue_observation_maps
 - compute_engage_observation_maps
 - compute_ally_observation_maps
 - compute_my_observation_maps
 
 - compute_blue_observation_maps_2
 - compute_ally_observation_maps_2
 - compute_my_observation_maps_2
 
 - compute_engage_mask_3
 - compute_blue_observation_maps_3
 - compute_engage_observation_maps_3
 - compute_ally_observation_maps_3
 - compute_my_observation_maps_3
 - compute_red_observation_maps_3 (For movie)
 
# For making result graph
 - make_test_results_graph_of_increase_number
"""


def summary_of_team(agents):
    # ef = efficiency x force
    ef = 0
    force = 0
    efficiency = 0

    num_platoons = 0
    num_companies = 0
    for agent in agents:
        ef += agent.ef
        force += agent.force
        efficiency += agent.efficiency

        if agent.type == 'platoon':
            num_platoons += 1
        else:
            num_companies += 1

    print('\n-------------------------------------------------------------')
    print(f'{agents[0].color} team: ', end='')
    print(f'num_agents:{num_platoons + num_companies},  ', end='')
    print(f'num_platoons:{num_platoons},  num_companies: {num_companies}')

    print(f'Total force = {force: .2f},  ', end='')
    print(f'Average force = {force / (num_platoons + num_companies): .2f},  ', end='')
    print(f'Average efficiency = {efficiency / (num_platoons + num_companies): .2f}')

    for agent in agents:
        print('(', agent.type, ',', end='')
        print(f'{agent.force: .2f}', ',', end='')
        print(f'{agent.efficiency:.2f}', '),', end='')

    print(f'\n Total eff x force={ef: .2f}')


def simulate_lanchester(r, b, config):
    x = np.array([config.R0, config.B0])
    matrix_a = np.array([[1, -b * config.dt], [-r * config.dt, 1]])

    max_rR = r * x[0]  # max of efficiency x force, rR_0
    max_bB = b * x[1]  # bB_0

    max_R = x[0]  # max of efficiency, R_0
    max_B = x[1]  # B_0

    history = {}
    t = 0.
    history['red_fe'] = np.array(r * x[0])  # rR_0
    history['blue_fe'] = np.array(b * x[1])  # bB_0
    history['red_force'] = np.array(x[0])  # R_0
    history['blue_force'] = np.array(x[1])  # B_0
    history['time'] = np.array(t)

    # print(f'\ntime:{t}, force:({x[0]:.1f}, {x[1]:.1f}) ', end='')
    # print(f'efficiency x force = ({r * x[0]:.1f}, {b * x[1]:.1f})')

    # Done criteria = efficiency x force > threshold
    while x[0] > config.threshold and x[1] > config.threshold:
        t += config.dt
        x = np.matmul(matrix_a, x)  # Lanchester model

        tmp = r * x[0] if r * x[0] > r * config.threshold else r * config.threshold
        history['red_fe'] = np.append(history['red_fe'], tmp)

        tmp = b * x[1] if b * x[1] > b * config.threshold else b * config.threshold
        history['blue_fe'] = np.append(history['blue_fe'], tmp)

        tmp = x[0] if x[0] > config.threshold else config.threshold
        history['red_force'] = np.append(history['red_force'], tmp)

        tmp = x[1] if x[1] > config.threshold else config.threshold
        history['blue_force'] = np.append(history['blue_force'], tmp)

        history['time'] = np.append(history['time'], t)

        # print(f'time:{t}, force:({x[0]:.1f}, {x[1]:.1f}) ', end='')
        # print(f'efficiency x force = ({r * x[0]:.1f}, {b * x[1]:.1f})')

    if history['red_force'][-1] == config.threshold:
        history['win'] = 'blue'
        history['lose'] = 'red'
    elif history['blue_force'][-1] == config.threshold:
        history['win'] = 'red'
        history['lose'] = 'blue'
    else:
        history['win'] = 'draw'
        history['lose'] = 'draw'

    # Efficiency
    red_efficiency = history['red_fe'] / history['red_force']
    blue_efficiency = history['blue_fe'] / history['blue_force']

    history['red_efficiency'] = red_efficiency
    history["blue_efficiency"] = blue_efficiency

    # Normalize force
    log_threshold = config.log_threshold
    log_n_max = np.max([config.log_R0, config.log_B0])
    denominator = log_n_max - log_threshold

    log_normalize_red_force = \
        compute_log_normalized_force(history['red_force'], log_threshold, denominator)
    log_normalize_blue_force = \
        compute_log_normalized_force(history['blue_force'], log_threshold, denominator)

    history['log_normalize_red_force'] = log_normalize_red_force
    history["log_normalize_blue_force"] = log_normalize_blue_force

    return history


def plot_force_history(history, r, b, config, dir_save):
    """
    This code is for 'simulate_lanchester'
    """
    plt.plot(history['time'], history['log_normalize_red_force'], marker="o", color='red')
    plt.plot(history['time'], history['log_normalize_blue_force'], marker="s", color='blue')
    plt.title(
        f'(R0, r)=({config.R0: .2f}, {r:.2f}); (B0,b)=({config.B0:.2f}, {b:.3f}); '
        f'threshold={config.threshold}')
    plt.ylabel('Normalized force')

    plt.xlabel('time')
    plt.grid()

    file_name = "force(" + str(config.R0) + "," + str(np.round(r, 2)) + ")-(" + \
                str(config.B0) + "," + str(np.round(b, 2)) + ")-" + str(config.threshold)
    plt.savefig(os.path.join(dir_save, "fig-" + file_name + ".png"))
    plt.show()

    # save config - TBD - エラーになる。
    # config_file = os.path.join(dir_save, "config-" + file_name + ".json")
    # with open(config_file, mode='wt', encoding='utf-8') as file:
    #    json.dump(config.__dict__, file)

    # save history
    history_file = os.path.join(dir_save, "history-" + file_name + ".pickle")
    with open(history_file, mode='wb') as file:  # 'wb' for byte
        pickle.dump(history, file)

    """ 参考: load pickle file """
    """
    with open(history_file, 'rb') as file:
        mydata = pickle.load(file)

    print(f'\nmydata ---------------------')
    print(mydata)
    """


def plot_efficiency_history(history, r, b, config, dir_save):
    """
    This code is for 'simulate_lanchester'
    """
    plt.plot(history['time'], history['red_efficiency'], marker="o", color='red')
    plt.plot(history['time'], history['blue_efficiency'], marker="s", color='blue')
    plt.title(
        f'(R0, r)=({config.R0:.2f}, {r:.2f}); (B0,b)=({config.B0:.2f}, {b:.2f}); '
        f'threshold={config.threshold}')
    plt.ylabel('Efficiency')

    plt.xlabel('time')
    plt.grid()

    file_name = "efficiency(" + str(config.R0) + "," + str(np.round(r, 2)) + ")-(" + \
                str(config.B0) + "," + str(np.round(b, 2)) + ")-" + str(config.threshold)
    plt.savefig(os.path.join(dir_save, "fig-" + file_name + ".png"))
    plt.show()

    # save history
    history_file = os.path.join(dir_save, "history-" + file_name + ".pickle")
    with open(history_file, mode='wb') as file:  # 'wb' for byte
        pickle.dump(history, file)


def visualize_blues_num_map(blues, battlefield, config):
    """
    This code is for debug.
        map of number of blue agents in the battlefield (monochrome)
    """

    blues_map = np.zeros(shape=(config.grid_size, config.grid_size))

    for blue in blues:
        blues_map[blue.pos[0], blue.pos[1]] += 1

    # plt.imshow(.5 * blues_map + battlefield)
    # plt.title(f'Allocation of blues, num_blues={len(blues)}')
    # plt.show()

    return blues_map


def visualize_reds_num_map(reds, battlefield, config):
    """
    This code is for debug.
        map of number of red agents in the battlefield (monochrome)
    """

    reds_map = np.zeros(shape=(config.grid_size, config.grid_size))

    for red in reds:
        reds_map[red.pos[0], red.pos[1]] += 1

    # plt.imshow(.5 * reds_map + battlefield)
    # plt.title(f'Allocation of reds, num_reds={len(reds)}')
    # plt.show()

    return reds_map


def visualize_battlefield_agents_num_map(battlefield, reds_map, blues_map, reds, blues, config,
                                         mode='humans'):
    """
    This code is for debug.
        map of number of red & blue agents in the battlefield (RGB)
    """

    print(f'reds_num_max/grid:{reds_map.max()}, blues_num_max/grid:{blues_map.max()}')

    rgb_r = np.expand_dims(reds_map, axis=2) / reds_map.max() + np.expand_dims(battlefield, axis=2)
    rgb_g = np.expand_dims(battlefield, axis=2)
    rgb_b = np.expand_dims(blues_map, axis=2) / blues_map.max() + np.expand_dims(battlefield,
                                                                                 axis=2)

    rgb_map = np.concatenate([rgb_r, rgb_g, rgb_b], axis=2)

    if mode == 'humans':
        # print(rgb_map.max())

        plt.figure(5)
        plt.imshow(rgb_map)
        plt.title(f'Initial Battle Fields: num_reds={len(reds)}, num_blues={len(blues)}')

        # file_name = "initial_battle_field.png"
        # file_name = "(" + str(config.R0) + "," + str(len(reds)) + ")-(" + \
        #            str(config.B0) + "," + str(len(blues)) + ")"
        # plt.savefig(os.path.join(dir_save, "initial_battlefield-" + file_name + ".png"))

        plt.show()

    return rgb_map


def compute_sum_of_rewards(rewards):
    return sum(rewards.values())


def count_alive_agents(agents):
    """ Count number of alive agents """
    alive_count = 0

    for agent in agents:
        if agent.alive:
            alive_count += 1

    return alive_count


def count_alive_platoons_and_companies(agents):
    """ Count number of alive platoons and companies """
    alive_platoons = 0
    alive_companies = 0

    for agent in agents:
        if agent.alive:
            if agent.type == 'platoon':
                alive_platoons += 1
            elif agent.type == 'company':
                alive_companies += 1
            else:
                raise NotImplementedError()

    return alive_platoons, alive_companies


def compute_current_total_ef_and_force(agents):
    """
    :param agnents: env.reds / env.blues
    """
    total_ef = 0.
    total_force = 0.

    total_effective_ef = 0.
    total_effective_force = 0.

    for agent in agents:
        if agent.alive:
            total_ef += agent.ef
            total_force += agent.force

            total_effective_ef += agent.effective_ef
            total_effective_force += agent.effective_force

    if (total_ef < 0) or (total_force < 0):
        raise ValueError()

    if (total_effective_ef < 0) or (total_effective_force < 0):
        raise ValueError()

    return total_ef, total_force, total_effective_ef, total_effective_force


def compute_log_normalized_force(force, log_threshold, denominator):
    numerator = np.log(force) - log_threshold
    log_normalized_force = numerator / denominator

    return log_normalized_force


def compute_engage_mask(env):
    """
    mask=1, if some reds & blues exist in the same cell
        red_ef, red_force: 2D map of reds, (grid_size,grid_size)
        blue_ef, blue_force: 2D map of blues, (grid_size,grid_size)
    """
    red_alive = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)
    blue_alive = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)

    red_ef = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)
    blue_ef = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)

    red_force = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)
    blue_force = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)

    for red in env.reds:
        if red.alive:
            red_alive[red.pos[0], red.pos[1]] = 1
            red_ef[red.pos[0], red.pos[1]] += red.ef
            red_force[red.pos[0], red.pos[1]] += red.force

    for blue in env.blues:
        if blue.alive:
            blue_alive[blue.pos[0], blue.pos[1]] = 1
            blue_ef[blue.pos[0], blue.pos[1]] += blue.ef
            blue_force[blue.pos[0], blue.pos[1]] += blue.force

    mask = red_alive * blue_alive  # masking engage cell

    return mask, red_ef, red_force, blue_ef, blue_force


def add_channel_dim(map_2D):
    return np.expand_dims(map_2D, axis=2).astype(np.float32)


def compute_log_normalized_cell_force(xf, yf, force, log_threshold, denominator, grid_size):
    # For observation
    log_normalized_force = np.zeros((grid_size, grid_size), dtype=np.float32)

    for (x, y) in zip(xf, yf):
        log_normalized_force[x, y] = \
            compute_log_normalized_force(force[x, y], log_threshold, denominator)

    # check
    if (np.any(log_normalized_force) < 0) or (np.any(log_normalized_force) > 1):
        raise ValueError()

    log_normalized_force = np.clip(log_normalized_force, a_min=0., a_max=1.)

    return log_normalized_force


def compute_cell_efficiency(xf, yf, force, ef, grid_size):
    # For observation
    efficiency = np.zeros((grid_size, grid_size), dtype=np.float32)

    for (x, y) in zip(xf, yf):
        efficiency[x, y] = ef[x, y] / force[x, y]

    # check
    if (np.any(efficiency) < 0) or (np.any(efficiency) > 1):
        raise ValueError()

    return efficiency


def compute_red_observation_maps(env):
    """
    Used in generate_movies.py
    Compute reds log normalized force and efficiency 2d maps.
    :return:
        red_log_normalized_force
        red_efficiency
    """

    red_ef = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)
    red_force = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)

    for red in env.reds:
        if red.alive:
            red_ef[red.pos[0], red.pos[1]] += red.ef
            red_force[red.pos[0], red.pos[1]] += red.force

    xf, yf = np.nonzero(red_force)

    # compute log_normalized_force
    log_threshold = env.config.log_threshold
    log_n_max = np.max([env.config.log_R0, env.config.log_B0])
    denominator = log_n_max - log_threshold

    red_log_normalized_force = \
        compute_log_normalized_cell_force(xf, yf, red_force,
                                          log_threshold, denominator, env.config.grid_size)

    # compute efficiency
    red_efficiency = compute_cell_efficiency(xf, yf, red_force, red_ef, env.config.grid_size)

    return red_log_normalized_force, red_efficiency


def compute_blue_observation_maps(env):
    """
    Compute blues log normalized force and efficiency 2d maps.
    :return:
        blue_log_normalized_force
        blue_efficiency
    """

    blue_ef = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)
    blue_force = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)

    for blue in env.blues:
        if blue.alive:
            blue_ef[blue.pos[0], blue.pos[1]] += blue.ef
            blue_force[blue.pos[0], blue.pos[1]] += blue.force

    xf, yf = np.nonzero(blue_force)

    # compute log_normalized_force
    log_threshold = env.config.log_threshold
    log_n_max = np.max([env.config.log_R0, env.config.log_B0])
    denominator = log_n_max - log_threshold

    blue_log_normalized_force = \
        compute_log_normalized_cell_force(xf, yf, blue_force,
                                          log_threshold, denominator, env.config.grid_size)

    # compute efficiency
    blue_efficiency = compute_cell_efficiency(xf, yf, blue_force, blue_ef, env.config.grid_size)

    return blue_log_normalized_force, blue_efficiency


def compute_engage_observation_maps(env):
    """
    Compute engage cell log normalized force 2d map, including myself.
    :return:
        engage_log_normalized_force
    """
    # Get engage mask
    mask, red_ef, red_force, blue_ef, blue_force = compute_engage_mask(env)

    engage_force = (red_force + blue_force) * mask
    xf, yf = np.nonzero(engage_force)

    # compute log_normalized_force
    log_threshold = env.config.log_threshold
    log_n_max = np.max([env.config.log_R0, env.config.log_B0])
    denominator = log_n_max - log_threshold

    engage_log_normalized_force = \
        compute_log_normalized_cell_force(xf, yf, engage_force,
                                          log_threshold + np.log(2), denominator,
                                          env.config.grid_size)

    return engage_log_normalized_force


def compute_ally_observation_maps(red, env):
    """
    Compute allies log normalized force and efficiency 2d map, except myself.
    i: myself id, red: myself
    :return:
        ally_log_normalized_force
        ally_efficiency
    """

    # ally (reds) except myself
    ally_ef = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)
    ally_force = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)

    for ally in env.reds:
        if ally.alive and (ally.id != red.id):
            ally_ef[ally.pos[0], ally.pos[1]] += ally.ef
            ally_force[ally.pos[0], ally.pos[1]] += ally.force

    xf, yf = np.nonzero(ally_force)

    # compute log_normalized_force
    log_threshold = env.config.log_threshold
    log_n_max = np.max([env.config.log_R0, env.config.log_B0])
    denominator = log_n_max - log_threshold

    ally_log_normalized_force = \
        compute_log_normalized_cell_force(xf, yf, ally_force,
                                          log_threshold, denominator, env.config.grid_size)

    # compute efficiency
    ally_efficiency = compute_cell_efficiency(xf, yf, ally_force, ally_ef, env.config.grid_size)

    return ally_log_normalized_force, ally_efficiency


def compute_my_observation_maps(red, env):
    """
    Compute my log normalized force and efficiency 2d map.
    red: myself
    :return:
        my_log_normalized_force
        my_efficiency
    """

    my_ef = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)
    my_force = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)

    my_ef[red.pos[0], red.pos[1]] = red.ef
    my_force[red.pos[0], red.pos[1]] = red.force

    xf, yf = np.nonzero(my_force)

    # compute log_normalized_force
    log_threshold = env.config.log_threshold
    log_n_max = np.max([env.config.log_R0, env.config.log_B0])
    denominator = log_n_max - log_threshold

    my_log_normalized_force = \
        compute_log_normalized_cell_force(xf, yf, my_force,
                                          log_threshold, denominator, env.config.grid_size)

    # compute efficiency
    my_efficiency = compute_cell_efficiency(xf, yf, my_force, my_ef, env.config.grid_size)

    return my_log_normalized_force, my_efficiency


def compute_blue_observation_maps_2(env):
    """
    Compute blues log normalized force and efficiency 2d maps.
    :return:
        blue_log_normalized_force
        blue_efficiency
        blue position
    """

    blue_ef = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)
    blue_force = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)
    blue_pos = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)

    for blue in env.blues:
        if blue.alive:
            blue_ef[blue.pos[0], blue.pos[1]] += blue.ef
            blue_force[blue.pos[0], blue.pos[1]] += blue.force
            blue_pos[blue.pos[0], blue.pos[1]] = 1

    xf, yf = np.nonzero(blue_force)

    # compute log_normalized_force
    log_threshold = env.config.log_threshold
    log_n_max = np.max([env.config.log_R0, env.config.log_B0])
    denominator = log_n_max - log_threshold

    blue_log_normalized_force = \
        compute_log_normalized_cell_force(xf, yf, blue_force,
                                          log_threshold, denominator, env.config.grid_size)

    # compute efficiency
    blue_efficiency = compute_cell_efficiency(xf, yf, blue_force, blue_ef, env.config.grid_size)

    return blue_log_normalized_force, blue_efficiency, blue_pos


def compute_ally_observation_maps_2(red, env):
    """
    Compute allies log normalized force and efficiency 2d map, except myself.
    i: myself id, red: myself
    :return:
        ally_log_normalized_force
        ally_efficiency
        ally position
    """

    # ally (reds) except myself
    ally_ef = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)
    ally_force = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)
    ally_pos = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)

    for ally in env.reds:
        if ally.alive and (ally.id != red.id):
            ally_ef[ally.pos[0], ally.pos[1]] += ally.ef
            ally_force[ally.pos[0], ally.pos[1]] += ally.force
            ally_pos[ally.pos[0], ally.pos[1]] = 1

    xf, yf = np.nonzero(ally_force)

    # compute log_normalized_force
    log_threshold = env.config.log_threshold
    log_n_max = np.max([env.config.log_R0, env.config.log_B0])
    denominator = log_n_max - log_threshold

    ally_log_normalized_force = \
        compute_log_normalized_cell_force(xf, yf, ally_force,
                                          log_threshold, denominator, env.config.grid_size)

    # compute efficiency
    ally_efficiency = compute_cell_efficiency(xf, yf, ally_force, ally_ef, env.config.grid_size)

    return ally_log_normalized_force, ally_efficiency, ally_pos


def compute_my_observation_maps_2(red, env):
    """
    Compute my log normalized force and efficiency 2d map.
    red: myself
    :return:
        my_log_normalized_force
        my_efficiency
        my_position
    """

    my_ef = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)
    my_force = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)
    my_pos = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)

    my_ef[red.pos[0], red.pos[1]] = red.ef
    my_force[red.pos[0], red.pos[1]] = red.force
    my_pos[red.pos[0], red.pos[1]] = 1

    xf, yf = np.nonzero(my_force)

    # compute log_normalized_force
    log_threshold = env.config.log_threshold
    log_n_max = np.max([env.config.log_R0, env.config.log_B0])
    denominator = log_n_max - log_threshold

    my_log_normalized_force = \
        compute_log_normalized_cell_force(xf, yf, my_force,
                                          log_threshold, denominator, env.config.grid_size)

    # compute efficiency
    my_efficiency = compute_cell_efficiency(xf, yf, my_force, my_ef, env.config.grid_size)

    return my_log_normalized_force, my_efficiency, my_pos


def compute_engage_mask_3(env):
    """
    mask=1, if some reds & blues exist in the same cell
        red_ef, red_force: 2D map of reds, (grid_size,grid_size)
        blue_ef, blue_force: 2D map of blues, (grid_size,grid_size)
    """
    red_alive = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)
    blue_alive = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)

    red_ef = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)
    blue_ef = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)

    red_force = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)
    blue_force = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)

    for red in env.reds:
        if red.alive:
            red_alive[red.pos[0], red.pos[1]] = 1
            red_ef[red.pos[0], red.pos[1]] += red.ef
            red_force[red.pos[0], red.pos[1]] += red.effective_force

    for blue in env.blues:
        if blue.alive:
            blue_alive[blue.pos[0], blue.pos[1]] = 1
            blue_ef[blue.pos[0], blue.pos[1]] += blue.ef
            blue_force[blue.pos[0], blue.pos[1]] += blue.effective_force

    mask = red_alive * blue_alive  # masking engage cell

    return mask, red_ef, red_force, blue_ef, blue_force


def compute_blue_observation_maps_3(env):
    """
    Compute blues log normalized force and efficiency 2d maps.
    :return:
        blue_normalized_force
        blue_efficiency
    """

    blue_ef = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)
    blue_force = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)
    blue_effective_force = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)

    for blue in env.blues:
        if blue.alive:
            blue_ef[blue.pos[0], blue.pos[1]] += blue.ef
            blue_force[blue.pos[0], blue.pos[1]] += blue.force
            blue_effective_force[blue.pos[0], blue.pos[1]] += blue.effective_force

    xf, yf = np.nonzero(blue_force)

    # compute normalized_force  only for alive blue
    alpha = 50.0
    blue_normalized_force = 2.0 / np.pi * np.arctan(blue_force / alpha)

    # compute efficiency
    blue_efficiency = compute_cell_efficiency(xf, yf, blue_force, blue_ef, env.config.grid_size)

    return blue_normalized_force, blue_efficiency


def compute_engage_observation_maps_3(env):
    """
    Compute engage cell log normalized force 2d map, including myself.
    :return:
        engage_normalized_force
    """
    # Get engage mask
    mask, red_ef, red_force, blue_ef, blue_force = compute_engage_mask(env)

    engage_force = (red_force + blue_force) * mask

    # compute normalized_force
    alpha = 50.0 * 2
    engage_normalized_force = 2.0 / np.pi * np.arctan(engage_force / alpha)

    return engage_normalized_force


def compute_ally_observation_maps_3(red, env):
    """
    Compute allies log normalized force and efficiency 2d map, except myself.
    i: myself id, red: myself
    :return:
        ally_normalized_force
        ally_efficiency
    """

    # ally (reds) except myself
    ally_ef = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)
    ally_force = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)
    ally_effective_force = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)

    for ally in env.reds:
        if ally.alive and (ally.id != red.id):
            ally_ef[ally.pos[0], ally.pos[1]] += ally.ef
            ally_force[ally.pos[0], ally.pos[1]] += ally.force
            ally_effective_force[ally.pos[0], ally.pos[1]] += ally.effective_force

    xf, yf = np.nonzero(ally_force)

    # compute normalized_force
    alpha = 50.0
    ally_normalized_force = 2.0 / np.pi * np.arctan(ally_force / alpha)

    # compute efficiency
    ally_efficiency = compute_cell_efficiency(xf, yf, ally_force, ally_ef, env.config.grid_size)

    return ally_normalized_force, ally_efficiency


def compute_my_observation_maps_3(red, env):
    """
    Compute my log normalized force and efficiency 2d map.
    red: myself
    :return:
        my_normalized_force
        my_efficiency
    """

    my_ef = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)
    my_force = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)
    my_effective_force = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)

    my_ef[red.pos[0], red.pos[1]] = red.ef
    my_force[red.pos[0], red.pos[1]] = red.force
    my_effective_force[red.pos[0], red.pos[1]] = red.effective_force

    xf, yf = np.nonzero(my_force)

    # compute log_normalized_force
    alpha = 50.0
    my_normalized_force = 2.0 / np.pi * np.arctan(my_force / alpha)

    # compute efficiency
    my_efficiency = compute_cell_efficiency(xf, yf, my_force, my_ef, env.config.grid_size)

    return my_normalized_force, my_efficiency


def compute_red_observation_maps_3(env):
    """
    Compute reds log normalized force and efficiency 2d maps.
    :return:
        red_normalized_force
        red_efficiency
    """

    red_ef = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)
    red_force = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)
    red_effective_force = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)

    for red in env.reds:
        if red.alive:
            red_ef[red.pos[0], red.pos[1]] += red.ef
            red_force[red.pos[0], red.pos[1]] += red.force
            red_effective_force[red.pos[0], red.pos[1]] += red.effective_force

    xf, yf = np.nonzero(red_force)

    # compute normalized_force  only for alive red
    alpha = 50.0
    red_normalized_force = 2.0 / np.pi * np.arctan(red_force / alpha)

    # compute efficiency
    red_efficiency = compute_cell_efficiency(xf, yf, red_force, red_ef, env.config.grid_size)

    return red_normalized_force, red_efficiency


def make_test_results_graph_of_increase_number(agent_type):
    num_red_win_list = []
    num_blue_win_list = []
    num_no_contest_list = []

    num_alive_reds_ratio_list = []
    num_alive_blues_ratio_list = []

    remaining_red_effective_force_ratio_list = []
    remaining_blue_effective_force_ratio_list = []

    episode_rewards_list = []
    episode_lens_list = []

    with open('trial-1/test_engagement/result_1000.json') as f:
        json_data = json.load(f)

        num_red_win_list.append(json_data['num_red_win'] / 1000)
        num_blue_win_list.append(json_data['num_blue_win'] / 1000)
        num_no_contest_list.append(json_data['no_contest'] / 1000)

        num_alive_reds_ratio_list.append(json_data['num_alive_reds_ratio'])
        num_alive_blues_ratio_list.append(json_data['num_alive_blues_ratio'])

        remaining_red_effective_force_ratio_list. \
            append(json_data['remaining_red_effective_force_ratio'])
        remaining_blue_effective_force_ratio_list. \
            append(json_data['remaining_blue_effective_force_ratio'])

        episode_rewards_list.append(json_data['episode_rewards'])
        episode_lens_list.append(json_data['episode_lens'])

    # agent_type = 'platoons', 'companies'
    parent_dir_1 = 'trial-1' + '/test_engagement/'
    parent_dir = parent_dir_1 + 'change_num_of_' + agent_type + '/'

    if agent_type == 'platoons' or agent_type == 'blue_platoons' or agent_type == 'red_platoons':
        file_dir = ['(11,20)', '(21,30)', '(31,40)', '(41,50)']
    elif agent_type == 'companies':
        file_dir = ['(6,10)', '(11,20)', '(21,30)', '(31,40)', '(41,50)']
    else:
        raise NotImplementedError()

    for file_name in file_dir:
        child_dir = agent_type + '=' + file_name + '/result.json'

        with open(parent_dir + child_dir, 'r') as f:
            json_data = json.load(f)

            num_red_win_list.append(json_data['num_red_win'] / 1000)
            num_blue_win_list.append(json_data['num_blue_win'] / 1000)
            num_no_contest_list.append(json_data['no_contest'] / 1000)

            num_alive_reds_ratio_list.append(json_data['num_alive_reds_ratio'])
            num_alive_blues_ratio_list.append(json_data['num_alive_blues_ratio'])

            remaining_red_effective_force_ratio_list. \
                append(json_data['remaining_red_effective_force_ratio'])
            remaining_blue_effective_force_ratio_list. \
                append(json_data['remaining_blue_effective_force_ratio'])

            episode_rewards_list.append(json_data['episode_rewards'])
            episode_lens_list.append(json_data['episode_lens'])

    savedir = Path(__file__).parent / parent_dir_1
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    if agent_type == 'platoons' or agent_type == 'blue_platoons' or agent_type == 'red_platoons':
        x = [6.5, 15.5, 25.5, 35.5, 45.5]
    elif agent_type == 'companies':
        x = [3.5, 8.0, 15.5, 25.5, 35.5, 45.5]
    else:
        NotImplementedError()

    plt.plot(x, num_red_win_list, color='r', marker='o', label='red win')
    plt.plot(x, num_blue_win_list, color='b', marker='o', label='blue win')
    plt.plot(x, num_no_contest_list, color='g', marker='s', label='no contest')
    plt.title('red win / blue win / no contest ratio, when increase num of ' + agent_type)
    plt.xlabel('num (bin range center) of ' + agent_type)
    plt.ylabel('win ratio')
    plt.ylim(-0.05, 1.05)
    plt.minorticks_on()
    plt.legend()
    plt.grid()

    savename = 'win_ratio_of_increase_number-'+ agent_type
    plt.savefig(str(savedir) + '/' + savename + '.png', dpi=300)
    plt.show()

    plt.plot(x, num_alive_reds_ratio_list, color='r', marker='o', label='alive red')
    plt.plot(x, num_alive_blues_ratio_list, color='b', marker='o', label='alive blue')
    plt.title('num alive agents ratio, when increase num of ' + agent_type)
    plt.xlabel('num (bin range center) of ' + agent_type)
    plt.ylabel('alive agents ratio')
    plt.ylim(-0.05, 1.05)
    plt.minorticks_on()
    plt.legend()
    plt.grid()
    # plt.yscale('log')

    savename = 'alive_agents_ratio_of_increase_number-'+ agent_type
    plt.savefig(str(savedir) + '/' + savename + '.png', dpi=300)
    plt.show()

    plt.plot(x, remaining_red_effective_force_ratio_list,
             color='r', marker='o', label='reds force')
    plt.plot(x, remaining_blue_effective_force_ratio_list,
             color='b', marker='o', label='blues force')
    plt.title('total remaining effective force ratio, when increase num of ' + agent_type)
    plt.xlabel('num (bin range center) of ' + agent_type)
    plt.ylabel('total remaining force ratio')
    plt.ylim(-0.05, 1.05)
    plt.minorticks_on()
    plt.legend()
    plt.grid()
    # plt.yscale('log')

    savename = 'remaining_force_of_increase_number-' + agent_type
    plt.savefig(str(savedir) + '/' + savename + '.png', dpi=300)
    plt.show()

    plt.plot(x, episode_rewards_list, color='r', marker='o', label='average episode reward')
    plt.plot(x, episode_lens_list, color='b', marker='s', label='average episode length')
    plt.title('average rewards and length of episodes, when increase num of ' + agent_type)
    plt.xlabel('num (bin range center) of ' + agent_type)
    plt.ylabel('rewards / length')
    plt.minorticks_on()
    plt.legend()
    plt.grid()

    savename = 'rewards_length_of_increase_number-' + agent_type
    plt.savefig(str(savedir) + '/' + savename + '.png', dpi=300)
    plt.show()
