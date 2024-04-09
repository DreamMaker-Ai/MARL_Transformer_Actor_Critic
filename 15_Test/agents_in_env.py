class RED:
    def __init__(self, agent_type, config):
        self.color = 'red'
        self.type = agent_type
        self.threshold = config.threshold
        self.id = None
        self.efficiency = None

        # Initial value
        self.initial_force = None
        self.initial_ef = None  # efficiency x force

        self.initial_effective_force = None  # initial_force - threshold
        self.initial_effective_ef = None  # initial_ef - threshold * efficiency

        # Current value
        self.pos = None
        self.force = None
        self.ef = None  # efficiency x force
        self.alive = True

        self.effective_force = None  # force - threshold/
        self.effective_ef = None  # ef - threshold * efficiency


class BLUE:
    def __init__(self, agent_type, config):
        self.color = 'blue'
        self.type = agent_type
        self.threshold = config.threshold
        self.id = None
        self.efficiency = None

        # Initial value
        self.initial_force = None
        self.initial_ef = None  # efficiency x force

        self.initial_effective_force = None  # initial_force - threshold
        self.initial_effective_ef = None  # initial_ef - threshold * efficiency

        # Current value
        self.pos = None
        self.force = None
        self.ef = None  # efficiency x force
        self.alive = True

        self.effective_force = None  # force - threshold
        self.effective_ef = None  # ef - threshold * efficiency


class BLOCK:
    def __init__(self):
        self.pos = None


def main():
    config = Config()
    red = RED('platoon', config=config)


if __name__ == '__main__':
    from config_for_test import Config

    main()
