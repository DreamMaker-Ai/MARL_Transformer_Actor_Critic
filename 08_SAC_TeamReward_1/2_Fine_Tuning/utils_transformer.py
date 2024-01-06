import numpy as np


def make_mask(alive_agents_ids, max_num_agents):
    mask = np.zeros(max_num_agents)  # (n,)

    for i in alive_agents_ids:
        mask[i] = 1

    mask = mask.astype(bool)  # (n,)
    mask = np.expand_dims(mask, axis=0)  # add batch_dim, (1,n)

    return mask


def make_padded_obs(max_num_agents, obs_shape, raw_obs):
    padding = np.zeros(obs_shape)  # 0-padding of obs
    padded_obs = []

    for i in range(max_num_agents):
        agent_id = 'red_' + str(i)

        if agent_id in raw_obs.keys():
            padded_obs.append(raw_obs[agent_id])
        else:
            padded_obs.append(padding)

    padded_obs = np.stack(padded_obs)  # stack to sequence (agent) dim  (n,g,g,ch*n_frames)

    padded_obs = np.expand_dims(padded_obs, axis=0)  # add batch_dim (1,n,g,g,ch*n_frames)

    return padded_obs


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
    print(np.sum(next_states, axis=(2,3,4)))

    max_num_agents = 10
    mask = make_mask(alive_agents_ids, max_num_agents)
    print(mask)


if __name__ == '__main__':
    main()
