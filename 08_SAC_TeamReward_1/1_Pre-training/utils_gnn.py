import re

import numpy as np
import tensorflow as tf

"""
adjacency_matrices
get_agents_states
get_agents_prev_actions (TODO)
get_agents_prev_actions_onehot (TODO)
get_alive_agents_ids
make_mask
make_reversal_mask_mat
"""


def adjacency_matrices(num_agents, alive_agents_ids):
    """
    Return list of adjacency matrix/zeros

    num_agents: max_num_red_agents
    alive_agents_ids: list[1,4,...]

    Return adjs; list[adj_0, adj_1, ...], len=num_agents
           adj_i;   if adj_i alive:
                        adjacency matrix of agnet_i; tf.Tensor(tf.float32), (n,n)
                    else:
                        zero; tf.Tensor(tf.float32), (n,n)
    """

    adjs = []

    adj_base = np.zeros(shape=(num_agents, num_agents), dtype=np.float32)
    for i in alive_agents_ids:
        adj_base[i, i] = 1.0

    for i in range(num_agents):
        if i in alive_agents_ids:
            myself = adj_base[i].reshape(1, num_agents)  # (1,num_agents)
            rem1 = adj_base[:i].reshape(-1, num_agents)  # (#,num_agents)
            rem2 = adj_base[i + 1:].reshape(-1, num_agents)  # (#,num_agents)

            adj = tf.concat([myself, rem1, rem2], axis=0)  # (num_agents, num_agents)

        else:
            adj = np.zeros(shape=(num_agents, num_agents), dtype=np.float32)

        adjs.append(tf.convert_to_tensor(adj, dtype=tf.float32))  # add (num_agents,num_agents)

    return adjs


def get_agents_states(env, states):
    """
    gent_states=[(grid,gird,ch*n_frames),...], len=max_num_red_agents
    """

    zero_states = tf.zeros((env.config.grid_size,
                            env.config.grid_size,
                            env.config.observation_channels * env.config.n_frames),
                           dtype=tf.float32)

    agents_states = []

    for red in env.reds:
        if red.alive:
            agents_states.append(tf.convert_to_tensor(states[red.id], dtype=tf.float32))
        else:
            agents_states.append(zero_states)

    for _ in range(env.config.num_red_agents, env.config.max_num_red_agents):
        agents_states.append(zero_states)

    return agents_states


def get_agents_prev_actions(env, prev_actions):
    # TODO
    """ dict -> list """
    zero_action = np.array([0])  # shape:(1,)
    agents_prev_actions = []

    for red in env.reds:
        if red.alive:
            prev_action = prev_actions[red.id]
            agents_prev_actions.append(prev_action)
        else:
            agents_prev_actions.append(zero_action)

    for _ in range(env.config.num_red_agents, env.config.max_num_red_agents):
        agents_prev_actions.append(zero_action)

    return agents_prev_actions


def get_agents_prev_actions_onehot(env, prev_actions):
    # TODO
    # prev_actions: [(b,1), ...], len=max_num_red_agents
    # To be implemented

    return None


def get_alive_agents_ids(env):
    alive_agents_ids = []

    for red in env.reds:
        if red.alive:
            alive_agents_ids.append(int(re.sub(r"\D", "", red.id)))

    return alive_agents_ids  # len=num_alive_red_agents


def make_mask(depth, alive_agents_ids):
    """
    :param depth: max_num_red_agents = n
    :param alive_agents_ids: [int,...], 可変長list
    :return: mask: (depth,), ndarray
    """

    x = np.eye(depth, dtype=np.float32)

    mask = np.sum(x[alive_agents_ids], axis=0)

    return mask


def make_reversal_mask_mat(mask, i):
    # Make reversal mask for agent_i attention score
    # mask[k, j] = 0 if agent k, j are alive
    # mask[k, j] = 1 otherwise

    # permute agents
    temp = mask[i]
    rem1 = mask[:i]
    rem2 = mask[i + 1:]
    mask = np.hstack([temp, rem1, rem2])

    # make mask=mat
    mask = np.expand_dims(mask, axis=0)

    mask_mat = np.matmul(mask.transpose(), mask)

    reversal_mask_mat = 1. - mask_mat

    return reversal_mask_mat, mask_mat, mask


def main():
    num_agents = 4
    alive_agents_ids = [0, 2]

    adjs = adjacency_matrices(num_agents, alive_agents_ids)

    mask = make_mask(num_agents, alive_agents_ids)
    print(f'mask={mask}, shape={mask.shape}')

    reversal_mask_mat_i, mask_mat_i, mask_i = make_reversal_mask_mat(mask=mask, i=2)
    print(f'mask_i={mask_i}, \n')
    print(f'mask_mat_i={mask_mat_i}, \n')
    print(f'reversal_mask_mat_i={reversal_mask_mat_i}')


if __name__ == '__main__':
    main()
