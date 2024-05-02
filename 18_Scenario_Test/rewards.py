import numpy as np


def get_rewards_before_engagement(reds_in_cell, blues_in_cell,
                                  reds_in_cell_ef, reds_in_cell_force,
                                  blues_in_cell_ef, blues_in_cell_force,
                                  env, rewards, infos):
    """
    call from engage
    Rewards based on local consolidation of force (magnification of red/blue force)
    """

    for red in reds_in_cell:
        if red.alive:

            if (reds_in_cell_force > blues_in_cell_force) and (reds_in_cell_ef > blues_in_cell_ef):
                rewards[red.id] += 2.0 / np.pi * \
                                   np.arctan(reds_in_cell_force / blues_in_cell_force) * \
                                   np.arctan(reds_in_cell_ef / blues_in_cell_ef)  # [0.5,1)

            else:
                # rewards[red.id] += -0.5  # For trial-0
                rewards[red.id] += 0.1  # For trial-1

            infos[str(red.pos) + ' ' + red.id]['raw_reward'] = np.round(rewards[red.id], 1)
            infos[str(red.pos) + ' ' + red.id]['reward'] = np.round(rewards[red.id], 1)

    return rewards, infos


def get_consolidation_of_force_rewards(env, rewards):
    """
    Called from step() in battlefield_strategy_for_test.py (not used)
    """
    beta = 0.2
    coef = 0.04
    reds = []

    for red in env.reds:
        if red.alive:
            reds.append(red)

    if len(reds) >= 2:
        for red_i in reds:

            r2 = 0.0
            for red_j in reds:
                r2 += (red_i.pos[0] - red_j.pos[0]) ** 2 + (red_i.pos[1] - red_j.pos[1]) ** 2

            r_i = np.sqrt(r2 / len(reds))  # rms

            if r_i > 0:
                rewards[red_i.id] += \
                    2.0 / np.pi * np.arctan(1.0 / (beta * r_i)) * coef  # (0,0.04)
            else:  # Consolidate to one force
                rewards[red_i.id] += 1.0 * coef  # 0.04

    elif len(reds) == 1:  # Only one alive
        for red in reds:
            rewards[red.id] += 1.0 * coef  # 0.04

    else:  # len(reds) == 0
        pass

    return rewards


def get_economy_of_force_rewards(env, rewards):
    """
    Called from step() in battlefield_strategy_for_test.py (not used)
    """
    beta = 0.1
    coef = 0.04

    reds = []
    blues = []
    blues_force = []

    for red in env.reds:
        if red.alive:
            reds.append(red)

    for blue in env.blues:
        if blue.alive:
            blues.append(blue)
            blues_force.append(blue.force)

    max_blue_id = np.argmax(blues_force)
    max_blue = blues[max_blue_id]

    for red in reds:
        r2 = (red.pos[0] - max_blue.pos[0]) ** 2 + (red.pos[1] - max_blue.pos[1]) ** 2
        r = np.sqrt(r2)

        if r > 0:
            rewards[red.id] += 2.0 / np.pi * np.arctan(1.0 / (beta * r)) * coef  # (0,0.04)
        else:
            rewards[red.id] += 1.0 * coef  # 0.04

    return rewards


def get_rewards_after_engagement(reds_in_cell, blues_in_cell, env, rewards, infos):
    """
    At cell(x,y), rewards based on the engagement result  (not used)
    - reds_in_cell, blues_in_cell: Alive agents list before engagement
    - blue.force: Blue agent force after angagement
    """
    blues_force = 0.0
    for blue in blues_in_cell:
        blues_force += blue.force

    if blues_force <= env.config.threshold * 1.001:
        for red in reds_in_cell:
            rewards[red.id] += 2.0

    return rewards, infos
