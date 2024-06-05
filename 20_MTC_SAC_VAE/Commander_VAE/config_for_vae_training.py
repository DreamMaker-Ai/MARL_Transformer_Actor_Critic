import numpy as np
import gym
import pickle


class Config:
    def __init__(self):
        self.batch_size = 32
        self.hidden_dim = 64
        self.learning_rate = 1e-4

        self.grid_size = None
        self.grid_range = (10, 50)
        self.offset = 0  # blue-team offset from edges

        self.global_grid_size = None

        self.commander_observation_channels = 6
        self.commander_n_frames = 1  # Default=1, assume no frame_stack
        self.commander_grid_size = 25
        self.latent_mult = 2  # lattent_dim = hidden_dim * latent_mult

        # Define Lanchester simulation parameters
        self.threshold = 5.0  # min of forces R & B
        self.log_threshold = np.log(self.threshold)
        self.mul = 2.0  # Minimum platoon force = threshold * mul

        # Define possible agent parameters
        self.agent_types = ('platoon', 'company')
        self.agent_forces = (50, 150)

        self.red_platoons = None
        self.red_companies = None
        self.blue_platoons = None
        self.blue_companies = None

        self.efficiencies_red = (0.3, 0.5)  # range
        self.efficiencies_blue = (0.3, 0.5)

        self.max_num_red_agents = None
        self.max_num_blue_agents = None

        # Red team TBD parameters
        self.R0 = None  # initial total force, set in 'generate_red_team'
        self.log_R0 = None
        self.num_red_agents = None  # set in 'define_red_team'
        self.num_red_platoons = None
        self.num_red_companies = None

        # Blue team TBD parameters
        self.B0 = None  # initial total force, set in 'generate_blue_team'
        self.log_B0 = None
        self.num_blue_agents = None  # set in 'define_blue_team'
        self.num_blue_platoons = None
        self.num_blue_companies = None

    def define_battlefield(self):
        # Define environment parameters
        self.grid_size = np.random.randint(low=self.grid_range[0], high=self.grid_range[1])

        self.global_grid_size = self.grid_size

        # Define possible red / blue agent parameters
        self.red_platoons = (0, int(self.grid_size * self.grid_size / 20))
        self.red_companies = (1, int(self.grid_size * self.grid_size / 20))

        self.blue_platoons = (0, int(self.grid_size * self.grid_size / 20))
        self.blue_companies = (1, int(self.grid_size * self.grid_size / 20))

        # For paddiing of multi-agents, *3 for adding red agents, default:*1
        self.max_num_red_agents = (self.red_platoons[1] + self.red_companies[1])
        self.max_num_blue_agents = (self.blue_platoons[1] + self.blue_companies[1])

    def define_blue_team(self):
        """
        Called from reset
            self.num_blue_agents, self.num_blue_platoons, self.num_blue_companies
            will be allocated.
        """
        self.num_blue_platoons = \
            np.random.randint(
                low=self.blue_platoons[0],
                high=self.blue_platoons[1] + 1)

        self.num_blue_companies = \
            np.random.randint(
                low=self.blue_companies[0],
                high=self.blue_companies[1] + 1)

        self.num_blue_agents = self.num_blue_platoons + self.num_blue_companies

    def define_red_team(self):
        """
        Called from reset
            self.num_red_agents, self.num_red_platoons, self.num_red_companies
            will be allocated.
        """
        self.num_red_platoons = \
            np.random.randint(
                low=self.red_platoons[0],
                high=self.red_platoons[1] + 1)

        self.num_red_companies = \
            np.random.randint(
                low=self.red_companies[0],
                high=self.red_companies[1] + 1)

        self.num_red_agents = self.num_red_platoons + self.num_red_companies

    def reset(self):
        """
        Generate new config for new episode
        """
        self.define_battlefield()
        self.define_blue_team()
        self.define_red_team()


if __name__ == '__main__':
    config = Config()

    for _ in range(3):
        config.reset()
