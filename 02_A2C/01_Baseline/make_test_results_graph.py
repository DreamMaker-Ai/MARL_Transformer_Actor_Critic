import os
import matplotlib.pyplot as plt
import json
from pathlib import Path


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

    parent_dir = 'trial' + '/robustness/'

    if agent_type == 'platoons' or agent_type == 'blue_platoons' or agent_type == 'red_platoons':
        file_dir = ['(3,10)', '(11,20)', '(21,30)', '(31,40)', '(41,50)']
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

    savedir = Path(__file__).parent / parent_dir
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

    savename = 'win_ratio_of_increase_number-' + agent_type
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

    savename = 'alive_agents_ratio_of_increase_number-' + agent_type
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


def main():
    """ Select one of followings. """
    # agent_type = 'platoons'
    # agent_type = 'companies'
    agent_type = 'blue_platoons'

    make_test_results_graph_of_increase_number(agent_type)


if __name__ == '__main__':
    main()
