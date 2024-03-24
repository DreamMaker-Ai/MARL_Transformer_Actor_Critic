import numpy as np

from utils import compute_engage_mask, compute_current_total_ef_and_force
from rewards import get_rewards_before_engagement, get_rewards_after_engagement


def set_info_red(env, red, infos):
    """ for each step """
    infos[str(red.pos) + ' ' + red.id] = {'time': np.round(env.step_count * env.config.dt, 1),
                                          'type': red.type,
                                          'efficiency': np.round(red.efficiency, 2),
                                          'ef': np.round(red.ef, 2),
                                          'force': np.round(red.force, 2),
                                          'next_ef': None,
                                          'next_force': None,
                                          'reward': None,
                                          'reward_scaling': None,
                                          'alive': None}

    return infos


def set_info_blue(env, blue, infos):
    """ for each step """
    infos[str(blue.pos) + ' ' + blue.id] = {'time': np.round(env.step_count * env.config.dt, 1),
                                            'type': blue.type,
                                            'efficiency': np.round(blue.efficiency, 2),
                                            'ef': np.round(blue.ef, 2),
                                            'force': np.round(blue.force, 2),
                                            'next_ef': None,
                                            'next_force': None,
                                            'alive': None}

    return infos


def add_info_next_ef_and_force(infos, agent):
    """
    (efficiency x force) and force after one step of Lanchester
    """
    infos[str(agent.pos) + ' ' + agent.id]['next_ef'] = np.round(agent.ef, 2)
    infos[str(agent.pos) + ' ' + agent.id]['next_force'] = np.round(agent.force, 2)

    return infos


def engage_and_get_rewards(env, x1, y1, rewards, infos):
    """
    Call from step()
    (x1,y1) ~ locations of engagement
    After one-step Lanchester simulation, red.ef, red.force, red.effective_ef, red.effective_force,
    blue.ef, blue.force, blue.effective_ef, blue.effective_force are updated.
    """

    for x, y in zip(x1, y1):
        """ before engage """
        # collect reds and blues in the cell
        reds_in_cell = []
        blues_in_cell = []

        for red in env.reds:
            if red.alive and red.pos == [x, y]:
                reds_in_cell.append(red)

                infos = set_info_red(env, red, infos)

        for blue in env.blues:
            if blue.alive and blue.pos == [x, y]:
                blues_in_cell.append(blue)

                infos = set_info_blue(env, blue, infos)

        if len(reds_in_cell) == 0 or len(blues_in_cell) == 0:
            raise ValueError()

        # Compute rR and R in the cell before engage (for Lanchester simulations)
        (reds_in_cell_ef, reds_in_cell_force, reds_in_cell_effective_ef,
         reds_in_cell_effective_force) = compute_current_total_ef_and_force(reds_in_cell)

        # Compute bB and b in the cell before engage (for Lanchester simulations)
        (blues_in_cell_ef, blues_in_cell_force, blues_in_cell_effective_ef,
         blues_in_cell_effective_force) = compute_current_total_ef_and_force(blues_in_cell)

        """
        Reward for consolidation of force 
        - Compute rewards of cell(x,y) based on before-engagement reds & blues status 
        """
        rewards, infos = \
            get_rewards_before_engagement(reds_in_cell, blues_in_cell,
                                          reds_in_cell_ef, reds_in_cell_force,
                                          blues_in_cell_ef, blues_in_cell_force,
                                          env, rewards, infos)

        """ engage (1 step of Lanchester simulation """
        # Update force of reds & blues in the cell
        # R_i' = R_i * (1 - bB / R * dt)
        next_red_force = []
        for red in reds_in_cell:
            next_red_force.append(
                max(red.force * (1 - blues_in_cell_ef / reds_in_cell_force * env.config.dt),
                    env.config.threshold))

        # B_i' = B_i * (1 - rR / B * dt)
        next_blue_force = []
        for blue in blues_in_cell:
            next_blue_force.append(
                max(blue.force * (1 - reds_in_cell_ef / blues_in_cell_force * env.config.dt),
                    env.config.threshold))

        """ after engage """
        # Update force, ef of reds & blues in the cell
        for i, red in enumerate(reds_in_cell):
            red.force = next_red_force[i]
            red.ef = red.force * red.efficiency

            red.effective_force = red.force - red.threshold
            red.effective_ef = red.ef - red.threshold * red.efficiency

            infos = add_info_next_ef_and_force(infos, red)

        for j, blue in enumerate(blues_in_cell):
            blue.force = next_blue_force[j]
            blue.ef = blue.force * blue.efficiency

            blue.effective_force = blue.force - blue.threshold
            blue.effective_ef = blue.ef - blue.threshold * blue.efficiency

            infos = add_info_next_ef_and_force(infos, blue)

        """
        Rewards after engagement
        - Compute rewards of cell(x,y) based on after-engagement reds & blues status
        """
        # rewards, infos = \
        #     get_rewards_after_engagement(reds_in_cell, blues_in_cell, env, rewards, infos)

    return rewards, infos


def get_dones(env, x1, y1, dones, infos):
    """ update agent alive and done of cell(x,y) after engagement """
    for x, y in zip(x1, y1):
        reds_in_cell = []  # alive agents list before engagement
        blues_in_cell = []

        for red in env.reds:
            if red.alive and red.pos == [x, y]:
                reds_in_cell.append(red)

        for blue in env.blues:
            if blue.alive and blue.pos == [x, y]:
                blues_in_cell.append(blue)

        dones, infos = get_dones_of_cell(env, reds_in_cell, blues_in_cell, dones, infos)

    return dones, infos


def get_dones_of_cell(env, reds_in_cell, blues_in_cell, dones, infos):
    """ evaluate done & alive of each agent """
    threshold = env.config.threshold * 1.001

    for red in reds_in_cell:
        if red.force <= threshold:
            dones[red.id] = True
            red.alive = False

            red.force = red.threshold
            red.ef = red.threshold * red.efficiency

            red.effective_ef = 0
            red.effective_force = 0

        infos[str(red.pos) + ' ' + red.id]['alive'] = red.alive

    for blue in blues_in_cell:
        if blue.force <= threshold:
            blue.alive = False

            blue.force = blue.threshold
            blue.ef = blue.threshold * blue.efficiency

            blue.effective_ef = 0
            blue.effective_force = 0

        infos[str(blue.pos) + ' ' + blue.id]['alive'] = blue.alive

    return dones, infos
