import numpy as np
import gym


class Config:
    def __init__(self):

        """ The model for the test is defined in tester.main. """
        # self.model_dir = 'models/model_215000/'  # newest file -> 'ls -ltr'
        self.model_dir = None

        # self.alpha_dir = 'models/alpha_215000.npy'  # logalpha
        self.alpha_dir = None

        if self.model_dir:  # starting steps for continual training
            self.n0 = 215000  # learner update cycles. Should be read from tensorboard
        else:
            self.n0 = 0

        # Define simulation cond.
        self.show_each_episode_result = True  # mainly for debug
        self.draw_win_distributions = False  # mainly for debug
        self.max_episodes_test_play = 1  # default=50 for training

        # Animation setting
        self.make_animation = True  # Use self.max_episodes_test_play=1

        # Time plot of a test setting
        self.make_time_plot = True  # Use self.max_episodes_test_play=1

        # POMDP params
        self.fov = 2  # default=2
        self.com = 8  # default=2

        # Define environment parameters
        self.grid_size = 50  # default=15
        self.offset = 0  # blue-team offset from edges

        # When agents are tested in larger battlefield, the global_model need to be build with
        # the grid_size at the training
        self.global_grid_size = 15

        # Define gym spaces
        self.action_dim = 5
        self.action_space = gym.spaces.Discrete(self.action_dim)

        observation_low = 0.
        observation_high = 1.
        self.observation_channels = 4  # default=4
        self.n_frames = 4  # default=4
        self.observation_space = \
            gym.spaces.Box(low=observation_low,
                           high=observation_high,
                           shape=(2 * self.fov + 1,
                                  2 * self.fov + 1,
                                  self.observation_channels)
                           )

        self.global_observation_channels = 6
        self.global_n_frames = 4

        # Replay buffer
        self.capacity = 10000  # Default=10000
        self.compress = True

        # Neural nets parameters
        self.hidden_dim = 64  # default=64
        self.key_dim = 64  # default=64
        self.num_heads = 2

        self.dropout_rate = 0.2  # default=0.2, (Dropout is not used.)

        # Training parameters
        self.worker_rollout_steps = 16  # default=16
        self.num_update_cycles = 100000000
        self.worker_rollouts_before_train = 50  # Default=50
        self.batch_size = 16  # default=16

        self.num_minibatchs = 3  # default=3
        self.tau = 0.01  # Soft update of target network
        self.gamma = 0.96
        self.max_steps = 400  # Default=100. 200 for robustness test

        self.learning_rate = 5e-5  # Default=5e-5
        self.alpha_learning_rate = 1e-5  # Default=1e-5

        self.ploss_coef = 0.1  # For policy_loss, Default=0.1
        self.aloss_coef = 0.1  # For entropy_loss, Default=0.1

        self.gradient_clip = 0.5  # clip_by_global_norm, Default=0.5
        self.alpha_clip = 0.5  # clip for alpha gradient, Default=0.5

        # Define Lanchester simulation parameters
        self.threshold = 5.0  # min of forces R & B
        self.log_threshold = np.log(self.threshold)
        self.mul = 2.0  # Minimum platoon force = threshold * mul
        self.dt = .2  # Default=.2

        # Define possible agent parameters
        self.agent_types = ('platoon', 'company')
        self.agent_forces = (50, 150)

        # Define possible agent parameters
        self.red_platoons = None
        self.red_companies = None
        self.red_pos = None
        self.blue_platoons = None
        self.blue_companies = None
        self.blue_pos = None
        self.efficiencies_red = None
        self.efficiencies_blue = None

        """ Define scenario 1-13 """
        self.scenario_id = 11
        self.read_test_scenario(self.scenario_id)

        # For paddiing of multi-agents, *3 for adding red agents, default:*1
        self.max_num_red_agents = (self.red_platoons[1] + self.red_companies[1])

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
        self.define_blue_team()
        self.define_red_team()

    def read_test_scenario(self, scenario_id):

        if scenario_id == 1:
            if self.grid_size != 15:
                raise ValueError('Wrong grid_size')

            self.red_platoons = (8, 8)  # num range of red platoons, default=(5,10)
            self.red_companies = (8, 8)  # num range of red companies, default=(5,10)
            self.red_pos = \
                [[1, 1], [7, 1], [13, 1], [1, 7], [13, 7], [1, 13], [7, 13], [13, 13],
                 [1, 2], [7, 2], [13, 2], [1, 8], [13, 8], [1, 12], [7, 12], [12, 12]]

            self.blue_platoons = (8, 8)  # num range of blue platoons, default=(5,10)
            self.blue_companies = (8, 8)  # num range of blue companies, default=(5,10)
            self.blue_pos = \
                [[5, 5], [7, 5], [9, 5], [6, 7], [8, 7], [5, 9], [7, 9], [9, 9],
                 [5, 6], [7, 6], [9, 6], [6, 8], [8, 8], [5, 10], [7, 10], [9, 10]]

            self.efficiencies_red = (0.4, 0.4)  # range
            self.efficiencies_blue = (0.4, 0.4)

        elif scenario_id == 2:
            if self.grid_size != 15:
                raise ValueError('Wrong grid_size')

            self.red_platoons = (8, 8)  # num range of red platoons, default=(5,10)
            self.red_companies = (8, 8)  # num range of red companies, default=(5,10)
            self.red_pos = \
                [[5, 5], [7, 5], [9, 5], [6, 7], [8, 7], [5, 9], [7, 9], [9, 9],
                 [5, 6], [7, 6], [9, 6], [6, 8], [8, 8], [5, 10], [7, 10], [9, 10]]

            self.blue_platoons = (8, 8)  # num range of blue platoons, default=(5,10)
            self.blue_companies = (8, 8)  # num range of blue companies, default=(5,10)
            self.blue_pos = \
                [[1, 1], [7, 1], [13, 1], [1, 7], [13, 7], [1, 13], [7, 13], [13, 13],
                 [1, 2], [7, 2], [13, 2], [1, 8], [13, 8], [1, 12], [7, 12], [12, 12]]

            self.efficiencies_red = (0.4, 0.4)  # range
            self.efficiencies_blue = (0.4, 0.4)

        elif scenario_id == 3:
            if self.grid_size != 15:
                raise ValueError('Wrong grid_size')

            self.red_platoons = (8, 8)  # num range of red platoons, default=(5,10)
            self.red_companies = (8, 8)  # num range of red companies, default=(5,10)
            self.red_pos = \
                [[1, 1], [1, 2], [5, 1], [5, 2], [9, 1], [9, 2], [12, 1], [12, 2],
                 [2, 1], [2, 2], [6, 1], [6, 2], [10, 1], [10, 2], [13, 1], [13, 2]]

            self.blue_platoons = (8, 8)  # num range of blue platoons, default=(5,10)
            self.blue_companies = (8, 8)  # num range of blue companies, default=(5,10)
            self.blue_pos = \
                [[1, 12], [1, 13], [5, 12], [5, 13], [9, 12], [9, 13], [12, 12], [12, 13],
                 [2, 12], [2, 13], [6, 12], [6, 13], [10, 12], [10, 13], [13, 12], [13, 13]]

            self.efficiencies_red = (0.4, 0.4)  # range
            self.efficiencies_blue = (0.4, 0.4)

        elif scenario_id == 4:
            if self.grid_size != 15:
                raise ValueError('Wrong grid_size')

            self.red_platoons = (8, 8)  # num range of red platoons, default=(5,10)
            self.red_companies = (8, 8)  # num range of red companies, default=(5,10)
            self.red_pos = \
                [[1, 12], [1, 13], [5, 12], [5, 13], [9, 12], [9, 13], [12, 12], [12, 13],
                 [2, 12], [2, 13], [6, 12], [6, 13], [10, 12], [10, 13], [13, 12], [13, 13]]

            self.blue_platoons = (8, 8)  # num range of blue platoons, default=(5,10)
            self.blue_companies = (8, 8)  # num range of blue companies, default=(5,10)
            self.blue_pos = \
                [[1, 1], [1, 2], [5, 1], [5, 2], [9, 1], [9, 2], [12, 1], [12, 2],
                 [2, 1], [2, 2], [6, 1], [6, 2], [10, 1], [10, 2], [13, 1], [13, 2]]

            self.efficiencies_red = (0.4, 0.4)  # range
            self.efficiencies_blue = (0.4, 0.4)

        elif scenario_id == 5:
            if self.grid_size != 15:
                raise ValueError('Wrong grid_size')

            self.red_platoons = (24, 24)
            self.red_companies = (20, 20)

            self.red_pos = []
            for i in range(7):
                if (i * 2 + 1 == 1) or (i * 2 + 1 == 13):
                    for j in range(7):
                        self.red_pos.append([i * 2 + 1, j * 2 + 1])
                else:
                    self.red_pos.append([i * 2 + 1, 1])
                    self.red_pos.append([i * 2 + 1, 13])

            for i in range(1, 7):
                if (i * 2 == 2) or (i * 2 == 12):
                    for j in range(1, 7):
                        self.red_pos.append([i * 2, j * 2])
                else:
                    self.red_pos.append([i * 2, 2])
                    self.red_pos.append([i * 2, 12])

            self.blue_platoons = (24, 24)
            self.blue_companies = (21, 21)

            self.blue_pos = []
            for i in [4, 10]:
                for j in [5, 6, 7, 8, 9]:
                    self.blue_pos.append([i, j])

            for i in [6, 8]:
                for j in [4, 5, 6, 7, 8, 9, 10]:
                    self.blue_pos.append([i, j])

            for i in [5, 7, 9]:
                for j in [4, 5, 6, 7, 8, 9, 10]:
                    self.blue_pos.append([i, j])

            self.efficiencies_red = (0.4, 0.4)  # range
            self.efficiencies_blue = (0.4, 0.4)

        elif scenario_id == 6:
            if self.grid_size != 15:
                raise ValueError('Wrong grid_size')

            self.blue_platoons = (24, 24)
            self.blue_companies = (20, 20)

            self.blue_pos = []
            for i in range(7):
                if (i * 2 + 1 == 1) or (i * 2 + 1 == 13):
                    for j in range(7):
                        self.blue_pos.append([i * 2 + 1, j * 2 + 1])
                else:
                    self.blue_pos.append([i * 2 + 1, 1])
                    self.blue_pos.append([i * 2 + 1, 13])

            for i in range(1, 7):
                if (i * 2 == 2) or (i * 2 == 12):
                    for j in range(1, 7):
                        self.blue_pos.append([i * 2, j * 2])
                else:
                    self.blue_pos.append([i * 2, 2])
                    self.blue_pos.append([i * 2, 12])

            self.red_platoons = (24, 24)
            self.red_companies = (21, 21)

            self.red_pos = []
            for i in [4, 10]:
                for j in [5, 6, 7, 8, 9]:
                    self.red_pos.append([i, j])

            for i in [6, 8]:
                for j in [4, 5, 6, 7, 8, 9, 10]:
                    self.red_pos.append([i, j])

            for i in [5, 7, 9]:
                for j in [4, 5, 6, 7, 8, 9, 10]:
                    self.red_pos.append([i, j])

            self.efficiencies_red = (0.4, 0.4)  # range
            self.efficiencies_blue = (0.4, 0.4)

        elif scenario_id == 7:
            if self.grid_size != 15:
                raise ValueError('Wrong grid_size')

            self.red_platoons = (12, 12)
            self.red_companies = (8, 8)

            self.red_pos = []

            # platoons
            for i in range(4):
                if (i * 4 + 1 == 1) or (i * 4 + 1 == 13):
                    for j in range(4):
                        self.red_pos.append([i * 4 + 1, j * 4 + 1])
                else:
                    self.red_pos.append([i * 4 + 1, 1])
                    self.red_pos.append([i * 4 + 1, 13])

            # companies
            for i in [2, 12]:
                for j in [3, 11]:
                    self.red_pos.append([i, j])

                self.red_pos.append([i, 7])

            self.red_pos.append([7, 2])
            self.red_pos.append([7, 12])

            self.blue_platoons = (9, 9)
            self.blue_companies = (12, 12)

            self.blue_pos = []
            # platoons
            for i in [5, 7, 9]:
                for j in [5, 7, 9]:
                    self.blue_pos.append([i, j])
            # companies
            for i in [5, 7, 9]:
                for j in [6, 8]:
                    self.blue_pos.append([i, j])

            for i in [6, 8]:
                for j in [5, 7, 9]:
                    self.blue_pos.append([i, j])

            self.efficiencies_red = (0.4, 0.4)  # range
            self.efficiencies_blue = (0.4, 0.4)

        elif scenario_id == 8:
            if self.grid_size != 15:
                raise ValueError('Wrong grid_size')

            self.blue_platoons = (12, 12)
            self.blue_companies = (8, 8)

            self.blue_pos = []

            # platoons
            for i in range(4):
                if (i * 4 + 1 == 1) or (i * 4 + 1 == 13):
                    for j in range(4):
                        self.blue_pos.append([i * 4 + 1, j * 4 + 1])
                else:
                    self.blue_pos.append([i * 4 + 1, 1])
                    self.blue_pos.append([i * 4 + 1, 13])

            # companies
            for i in [2, 12]:
                for j in [3, 11]:
                    self.blue_pos.append([i, j])

                self.blue_pos.append([i, 7])

            self.blue_pos.append([7, 2])
            self.blue_pos.append([7, 12])

            self.red_platoons = (9, 9)
            self.red_companies = (12, 12)

            self.red_pos = []
            # platoons
            for i in [5, 7, 9]:
                for j in [5, 7, 9]:
                    self.red_pos.append([i, j])
            # companies
            for i in [5, 7, 9]:
                for j in [6, 8]:
                    self.red_pos.append([i, j])

            for i in [6, 8]:
                for j in [5, 7, 9]:
                    self.red_pos.append([i, j])

            self.efficiencies_red = (0.4, 0.4)  # range
            self.efficiencies_blue = (0.4, 0.4)

        elif scenario_id == 9:
            if self.grid_size != 15:
                raise ValueError('Wrong grid_size')

            self.blue_platoons = (12, 12)
            self.blue_companies = (8, 8)

            self.blue_pos = []

            # platoons
            for i in range(4):
                if (i * 4 + 1 == 1) or (i * 4 + 1 == 13):
                    for j in range(4):
                        self.blue_pos.append([i * 4 + 1, j * 4 + 1])
                else:
                    self.blue_pos.append([i * 4 + 1, 1])
                    self.blue_pos.append([i * 4 + 1, 13])

            # companies
            for i in [2, 12]:
                for j in [3, 11]:
                    self.blue_pos.append([i, j])

                self.blue_pos.append([i, 7])

            self.blue_pos.append([7, 2])
            self.blue_pos.append([7, 12])

            self.red_platoons = (8, 8)
            self.red_companies = (5, 5)

            self.red_pos = []
            # platoons
            platoons_pos = [[5, 7], [6, 7], [7, 5], [7, 6], [7, 8], [7, 9], [8, 7], [9, 7]]

            for pos in platoons_pos:
                self.red_pos.append(pos)

            # companies
            companies_pos = [[6, 6], [8, 6], [7, 7], [6, 8], [8, 8]]
            for pos in companies_pos:
                self.red_pos.append(pos)

            self.efficiencies_red = (0.4, 0.4)  # range
            self.efficiencies_blue = (0.4, 0.4)

        elif scenario_id == 10:
            if self.grid_size != 50:
                raise ValueError('Wrong grid_size')

            # red_agents
            self.red_pos = []

            for x in range(1, 48, 2):
                for y in range(1, 7, 2):
                    self.red_pos.append([x, y])

            for x in range(1, 7, 2):
                for y in range(9, 40, 2):
                    self.red_pos.append([x, y])

            for x in range(43, 48, 2):
                for y in range(9, 40, 2):
                    self.red_pos.append([x, y])

            for x in range(1, 48, 2):
                for y in range(42, 48, 2):
                    self.red_pos.append([x, y])

            # blue_agents
            self.blue_pos = []

            for x in range(7, 42, 2):
                for y in range(8, 42, 2):
                    self.blue_pos.append([x, y])

            num_red_platoons = len(self.red_pos) // 2
            num_red_companies = len(self.red_pos) - num_red_platoons

            self.red_platoons = (num_red_platoons, num_red_platoons)
            self.red_companies = (num_red_companies, num_red_companies)

            num_blue_platoons = len(self.blue_pos) // 2
            num_blue_companies = len(self.blue_pos) - num_blue_platoons

            self.blue_platoons = (num_blue_platoons, num_blue_platoons)
            self.blue_companies = (num_blue_companies, num_blue_companies)
            
            self.efficiencies_red = (0.4, 0.4)  # range
            self.efficiencies_blue = (0.4, 0.4)

        elif scenario_id == 11:
            if self.grid_size != 50:
                raise ValueError('Wrong grid_size')

            # blue_agents
            self.blue_pos = []

            for x in range(1, 48, 2):
                for y in range(1, 7, 2):
                    self.blue_pos.append([x, y])

            for x in range(1, 7, 2):
                for y in range(9, 40, 2):
                    self.blue_pos.append([x, y])

            for x in range(43, 48, 2):
                for y in range(9, 40, 2):
                    self.blue_pos.append([x, y])

            for x in range(1, 48, 2):
                for y in range(42, 48, 2):
                    self.blue_pos.append([x, y])

            # red_agents
            self.red_pos = []

            for x in range(7, 42, 3):
                for y in range(8, 42, 3):
                    self.red_pos.append([x, y])

            num_red_platoons = len(self.red_pos) // 2
            num_red_companies = len(self.red_pos) - num_red_platoons

            self.red_platoons = (num_red_platoons, num_red_platoons)
            self.red_companies = (num_red_companies, num_red_companies)

            num_blue_platoons = len(self.blue_pos) // 2
            num_blue_companies = len(self.blue_pos) - num_blue_platoons

            self.blue_platoons = (num_blue_platoons, num_blue_platoons)
            self.blue_companies = (num_blue_companies, num_blue_companies)

            self.efficiencies_red = (0.4, 0.4)  # range
            self.efficiencies_blue = (0.4, 0.4)

        elif scenario_id == 12:
            if self.grid_size != 50:
                raise ValueError('Wrong grid_size')

            self.red_pos = []

            for x in range(1, 20, 2):
                for y in range(1, 49, 2):
                    self.red_pos.append([x, y])

            self.blue_pos = []

            for x in range(25, 49, 2):
                for y in range(1, 49, 2):
                    self.blue_pos.append([x, y])

            num_red_platoons = len(self.red_pos) // 2
            num_red_companies = len(self.red_pos) - num_red_platoons

            self.red_platoons = (num_red_platoons, num_red_platoons)
            self.red_companies = (num_red_companies, num_red_companies)

            num_blue_platoons = len(self.blue_pos) // 2
            num_blue_companies = len(self.blue_pos) - num_blue_platoons

            self.blue_platoons = (num_blue_platoons, num_blue_platoons)
            self.blue_companies = (num_blue_companies, num_blue_companies)

            self.efficiencies_red = (0.4, 0.4)  # range
            self.efficiencies_blue = (0.4, 0.4)

        elif scenario_id == 13:
            if self.grid_size != 50:
                raise ValueError('Wrong grid_size')

            self.red_pos = []

            for x in range(30, 49, 2):
                for y in range(1, 49, 2):
                    self.red_pos.append([x, y])

            self.blue_pos = []

            for x in range(1, 25, 2):
                for y in range(1, 49, 2):
                    self.blue_pos.append([x, y])

            num_red_platoons = len(self.red_pos) // 2
            num_red_companies = len(self.red_pos) - num_red_platoons

            self.red_platoons = (num_red_platoons, num_red_platoons)
            self.red_companies = (num_red_companies, num_red_companies)

            num_blue_platoons = len(self.blue_pos) // 2
            num_blue_companies = len(self.blue_pos) - num_blue_platoons

            self.blue_platoons = (num_blue_platoons, num_blue_platoons)
            self.blue_companies = (num_blue_companies, num_blue_companies)

            self.efficiencies_red = (0.4, 0.4)  # range
            self.efficiencies_blue = (0.4, 0.4)

        else:
            raise NotImplementedError()


if __name__ == '__main__':
    config = Config()

    for _ in range(3):
        config.reset()
