import numpy as np


def make_po_attention_mask(alive_agents_ids, max_num_agents, agents, com):
    """ attention mask (=adjacency matrix): (1,n,n), bool; For red and blue agents """

    attention_mask = np.zeros((max_num_agents, max_num_agents))  # (n,n)

    for i in alive_agents_ids:
        for j in alive_agents_ids:
            if (np.abs(agents[i].pos[0] - agents[j].pos[0]) <= com) \
                    and (np.abs(agents[i].pos[1] - agents[j].pos[1]) <= com):
                attention_mask[i, j] = 1

    attention_mask = attention_mask.astype(bool)

    attention_mask = np.expand_dims(attention_mask, axis=0)  # (1,n,n)

    return attention_mask


def make_mask(alive_agents_ids, max_num_agents):
    """ alive mask; For red and blue agents """

    mask = np.zeros(max_num_agents)  # (n,)

    for i in alive_agents_ids:
        mask[i] = 1

    mask = mask.astype(bool)  # (n,)
    mask = np.expand_dims(mask, axis=0)  # add batch_dim, (1,n)

    return mask


def make_padded_obs(max_num_agents, obs_shape, raw_obs):  # For red agents
    padding = np.zeros(obs_shape)  # 0-padding of obs, (5,5,16)
    padded_obs = []

    for i in range(max_num_agents):
        agent_id = 'red_' + str(i)

        if agent_id in raw_obs.keys():  # alive
            padded_obs.append(raw_obs[agent_id])
        else:  # dummy/dead
            padded_obs.append(padding)

    # stack to sequence (agent) dim  (n,2*fov+1,2*fov+1,ch*n_frames)=(15,5,5,16)
    padded_obs = np.stack(padded_obs)

    # add batch_dim (1,n,2*fov+1,2*fov+1,ch*n_frames)=(1,15,5,5,16)
    padded_obs = np.expand_dims(padded_obs, axis=0)

    return padded_obs


def make_blues_padded_obs(max_num_agents, obs_shape, raw_obs):  # For blue agents
    padding = np.zeros(obs_shape)  # 0-padding of obs, (5,5,16)
    padded_obs = []

    for i in range(max_num_agents):
        agent_id = 'blue_' + str(i)

        if agent_id in raw_obs.keys():  # alive
            padded_obs.append(raw_obs[agent_id])
        else:  # dummy/dead
            padded_obs.append(padding)

    # stack to sequence (agent) dim  (n,2*fov+1,2*fov+1,ch*n_frames)=(15,5,5,16)
    padded_obs = np.stack(padded_obs)

    # add batch_dim (1,n,2*fov+1,2*fov+1,ch*n_frames)=(1,15,5,5,16)
    padded_obs = np.expand_dims(padded_obs, axis=0)

    return padded_obs


def make_padded_pos(max_num_agents, pos_shape, raw_pos):  # For red agents
    padding = np.zeros(pos_shape)  # 0-padding of pos, (8,)
    padded_pos = []

    for i in range(max_num_agents):
        agent_id = 'red_' + str(i)

        if agent_id in raw_pos.keys():  # alive
            padded_pos.append(raw_pos[agent_id])
        else:  # dummy/dead
            padded_pos.append(padding)

    # stack to sequence (agent) dim  (n,2*n_frames)=(15,8)
    padded_pos = np.stack(padded_pos)

    # add batch_dim (1,n,2n_frames)=(1,15,8)
    padded_pos = np.expand_dims(padded_pos, axis=0)

    return padded_pos


def make_blues_padded_pos(max_num_agents, pos_shape, raw_pos):  # For blue agents
    padding = np.zeros(pos_shape)  # 0-padding of pos, (8,)
    padded_pos = []

    for i in range(max_num_agents):
        agent_id = 'blue_' + str(i)

        if agent_id in raw_pos.keys():  # alive
            padded_pos.append(raw_pos[agent_id])
        else:  # dummy/dead
            padded_pos.append(padding)

    # stack to sequence (agent) dim  (n,2*n_frames)=(15,8)
    padded_pos = np.stack(padded_pos)

    # add batch_dim (1,n,2n_frames)=(1,15,8)
    padded_pos = np.expand_dims(padded_pos, axis=0)

    return padded_pos


def main():
    alive_agents_ids = [0, 2, 3, 4, 5, 7]
    print(alive_agents_ids)
    obs_shape = (2, 2, 1)

    next_agent_states = {}
    for i in alive_agents_ids:
        agent_id = 'red_' + str(i)
        next_agent_states[agent_id] = np.ones(obs_shape)

    next_states = \
        make_padded_obs(
            max_num_agents=15,
            obs_shape=obs_shape,
            raw_obs=next_agent_states,
        )

    print(next_states.shape)
    print(np.sum(next_states, axis=(2, 3, 4)))

    max_num_agents = 10
    mask = make_mask(alive_agents_ids, max_num_agents)
    print(mask)


if __name__ == '__main__':
    main()
