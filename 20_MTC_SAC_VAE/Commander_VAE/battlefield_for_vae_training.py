import numpy as np

from agents_in_env import RED, BLUE
from config_for_vae_training import Config
from generate_agents_in_env import generate_red_team, generate_blue_team

from commander_observations_for_vae_training \
    import get_commander_observation, commander_state_resize

import matplotlib.pyplot as plt


class BattleFieldStrategy:
    def __init__(self):
        super(BattleFieldStrategy, self).__init__()

        self.config = Config()

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

        commander_observation = get_commander_observation(self)  # (g,g,6)

        return commander_observation

    def plot_commander_maps(self, fig_1, fig_2, fig_color):
        fig = plt.figure()
        x = 1
        y = 2

        implot1 = 1
        ax1 = fig.add_subplot(x, y, implot1)
        ax1.set_title("Original map", fontsize=20)
        plt.imshow(fig_1, cmap=fig_color)

        implot2 = 2
        ax2 = fig.add_subplot(x, y, implot2)
        ax2.set_title("Resized map", fontsize=20)
        plt.imshow(fig_2, cmap=fig_color)

        plt.show()

        pass


def main():
    env = BattleFieldStrategy()

    for _ in range(2):
        commander_observation = env.reset()  # (g,g,6)

        resized_commander_observation = \
            commander_state_resize(commander_observation, env.config.commander_grid_size)

        """ Reds forces """
        env.plot_commander_maps(commander_observation[:, :, 0],
                                resized_commander_observation[:, :, 0],
                                "Reds")
        plt.show()

        """ Blues forces """
        env.plot_commander_maps(commander_observation[:, :, 2],
                                resized_commander_observation[:, :, 2],
                                "Blues")
        plt.show()

        """ Reds sin position """
        env.plot_commander_maps(commander_observation[:, :, 4],
                                resized_commander_observation[:, :, 4],
                                "Greens")
        plt.show()

        """ Reds cos position """
        env.plot_commander_maps(commander_observation[:, :, 5],
                                resized_commander_observation[:, :, 5],
                                "Greens")
        plt.show()


if __name__ == "__main__":
    main()
