import numpy as np
import matplotlib.pyplot as plt

from config_finetuning import Config
# from generate_env import random_shape_maze
from agents_in_env import BLUE, RED
from utils import summary_of_team, simulate_lanchester, \
    plot_force_history, plot_efficiency_history, \
    visualize_blues_num_map, visualize_reds_num_map, visualize_battlefield_agents_num_map


def compute_initial_total_force_and_ef(agents):
    """ Compute initial forces R0, B0 of team (for checking purpose) """
    initial_force = 0
    initial_ef = 0

    initial_effective_force = 0
    initial_effective_ef = 0

    for agent in agents:
        initial_force += agent.initial_force
        initial_ef += agent.initial_ef

        initial_effective_force += agent.initial_effective_force
        initial_effective_ef += agent.initial_effective_ef

    return initial_force, initial_ef, initial_effective_force, initial_effective_ef


def check_obstacles(x, y, battlefield):
    if battlefield[x, y] == 0:
        obs_exists = False
    else:
        obs_exists = True

    return obs_exists


def check_blues(x, y, blues):
    blues_exist = False

    for blue in blues:
        if (blue.pos[0] == x) and (blue.pos[1] == y):
            blues_exist = True

    return blues_exist


def generate_blue_team(agent_class, config, battlefield):
    """
    Call this for generating the blue team
    Minimum force = threshold * mul, mul=2.0 (default)
    """

    if config.threshold * config.mul > config.agent_forces[0]:
        raise ValueError()

    num_agents = config.num_blue_agents
    num_platoons = config.num_blue_platoons
    num_companies = config.num_blue_companies
    efficiencies = config.efficiencies_blue  # blues efficiency range

    agents = []
    for _ in range(num_platoons):
        agent = agent_class(agent_type='platoon', config=config)
        agent.force = 40.0 + 10.0 * (np.random.rand() - 0.5)
        agents.append(agent)

    for _ in range(num_companies):
        agent = agent_class(agent_type='company', config=config)
        agent.force = 135.0 + 20.0 * (np.random.rand() - 0.5)
        agents.append(agent)

    for idx, agent in enumerate(agents):
        agent.id = agent.color + '_' + str(idx)
        agent.efficiency = efficiencies[0] + (efficiencies[1] - efficiencies[0]) * np.random.rand()
        agent.ef = agent.force * agent.efficiency

        agent.effective_force = agent.force - agent.threshold
        agent.effective_ef = agent.ef - agent.threshold * agent.efficiency

        agent.initial_force = agent.force
        agent.initial_ef = agent.ef

        agent.initial_effective_force = agent.effective_force
        agent.initial_effective_ef = agent.effective_ef

    """ compute initial total force and (efficiency x force) """
    initial_force, initial_ef, initial_effective_force, initial_effective_ef = \
        compute_initial_total_force_and_ef(agents)

    config.B0 = initial_force
    config.log_B0 = np.log(initial_force)

    """ allocate agent's initial position """
    for i, agent in enumerate(agents):
        agent.pos = (config.blue_pos[i] +
                     np.random.choice([-1, 0, 1], size=2, p=[0.2, 0.6, 0.2])).tolist()

    return agents, initial_ef, initial_force, initial_effective_ef, initial_effective_force


def generate_red_team(agent_class, config, battlefield, blues):
    """
    Call this for generating the red team
    Minimum force = threshold * mul, mul=2.0 (default)
    """

    if config.threshold * config.mul > config.agent_forces[0]:
        raise ValueError()

    num_agents = config.num_red_agents
    num_platoons = config.num_red_platoons
    num_companies = config.num_red_companies
    efficiencies = config.efficiencies_red  # reds efficiency range

    agents = []
    for _ in range(num_platoons):
        agent = agent_class(agent_type='platoon', config=config)
        agent.force = 40.0 + 10.0 * (np.random.rand() - 0.5)
        agents.append(agent)

    for _ in range(num_companies):
        agent = agent_class(agent_type='company', config=config)
        agent.force = 135.0 + 20.0 * (np.random.rand() - 0.5)
        agents.append(agent)

    for idx, agent in enumerate(agents):
        agent.id = agent.color + '_' + str(idx)
        agent.efficiency = efficiencies[0] + (efficiencies[1] - efficiencies[0]) * np.random.rand()
        agent.ef = agent.force * agent.efficiency

        agent.effective_force = agent.force - agent.threshold
        agent.effective_ef = agent.ef - agent.threshold * agent.efficiency

        agent.initial_force = agent.force
        agent.initial_ef = agent.ef

        agent.initial_effective_force = agent.effective_force
        agent.initial_effective_ef = agent.effective_ef

    """ compute initial total force and (efficiency x force) """
    initial_force, initial_ef, initial_effective_force, initial_effective_ef = \
        compute_initial_total_force_and_ef(agents)

    config.R0 = initial_force
    config.log_R0 = np.log(initial_force)

    """ allocate agent's initial position (need blues) """

    for i, agent in enumerate(agents):
        agent.pos = (config.red_pos[i] +
                     np.random.choice([-1, 0, 1], size=2, p=[0.2, 0.6, 0.2])).tolist()

    return agents, initial_ef, initial_force, initial_effective_ef, initial_effective_force


def main():
    """ Define configuration """
    config = Config()
    config.reset()

    """ Generate battlefield """
    battlefield = np.zeros((config.grid_size, config.grid_size))

    plt.imshow(battlefield)
    plt.title('battle field')
    plt.show()

    """ Generate Blue team """
    blues, _, _, _, _ = generate_blue_team(BLUE, config, battlefield)

    summary_of_team(blues)

    blues_map = visualize_blues_num_map(blues, battlefield, config)

    """ Generate Red team """
    reds, _, _, _, _ = generate_red_team(RED, config, battlefield, blues)

    summary_of_team(reds)

    reds_map = visualize_reds_num_map(reds, battlefield, config)

    """ Visualize initial battlefield """
    visualize_battlefield_agents_num_map(battlefield, reds_map, blues_map, reds, blues, config)

    """ Lanchester simulations """
    # Compute average efficiency and total force
    R0, rR0, _, _ = compute_initial_total_force_and_ef(reds)
    average_r = rR0 / R0

    if R0 != config.R0:
        raise ValueError()

    B0, bB0, _, _ = compute_initial_total_force_and_ef(blues)
    average_b = bB0 / B0

    if B0 != config.B0:
        raise ValueError()

    history = simulate_lanchester(average_r, average_b, config)

    plot_force_history(history, average_r, average_b, config, dir_save)
    plot_efficiency_history(history, average_r, average_b, config, dir_save)


if __name__ == '__main__':
    import os

    dir_save = './simulate_lanchester_model_with_reds_and_blues'
    if not os.path.exists(dir_save):
        os.mkdir(dir_save)

    main()
