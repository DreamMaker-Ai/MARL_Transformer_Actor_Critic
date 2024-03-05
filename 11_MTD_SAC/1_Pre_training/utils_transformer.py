import numpy as np
import copy
import tensorflow as tf

"""
    Originally copied from "MT_APEX_DQN/12_MTD/MTD_rev"
    Replace "make_mask", "make_padded_obs" to "10_SAC_GlobalState_1"
"""


def make_id_mask(alive_agents_ids, max_num_agents):  # (batch,1,n)=(1,1,n)
    """ masks: [(1,1,n), ...], len=n=max_num_agents """

    masks = []
    for _ in range(max_num_agents):
        masks.append(np.zeros((1, 1, max_num_agents)))

    for i in alive_agents_ids:
        for j in alive_agents_ids:
            masks[i][:, :, j] = 1

    for i in range(max_num_agents):
        masks[i] = masks[i].astype(bool)

    return masks


def experiences2per_agent_list(self, experiences):
    """
    experiences には、batch (=16) 分の transition が入っている。

    :input:
        experience.obss = [obs of agent_1: (1,15,15,6),
                           obs of agent_2: (1,15,15,6),,
                              ... ], len=n
        experience.actions = [action of agent_1: (1,),
                              action of agent_2: (1,),
                              ... ], len=n
        experience.masks: [mask of agent_1= (1,1,15),
                           mask of agent_2= (1,1,15),
                              ... ], len=n, bool
        experience.global_r: (1,1)
        experience.global_state: (1,15,15,6)

    :return:
        agent_obs = [
        obs list of agent_1: [(1,g,g,ch*n_frames),...], len=b
        obs list of agent_2: [(1,g,g,ch*n_frames),...], len=b
            ...
        ], len=n

        agent_action = [
            action list of agent_1: [(1,),...], len=b
            action list of agent_2: [(1,),...], len=b
            ...
        ], len=n

        agent_mask = [
            mask list of agent_1: [(1,1,n),...], len=b
            mask list of agent_2: [(1,1,n),...], len=b
            ...
        ], len=n, bool

        team_reward: [(1,1),...], len=b
        team_done: [(1,1),...], len=b, bool

        global_state: [(1,g,g,global_ch*global_n_frames),...], len=b

        etc.
    """

    agent_obs = []
    agent_action = []
    agent_reward = []
    agent_next_obs = []
    agent_done = []
    team_r = []
    team_done = []
    agent_mask = []
    agent_next_mask = []
    global_state = []
    next_global_state = []

    for _ in range(self.env.config.max_num_red_agents):
        agent_obs.append([])
        agent_action.append([])
        agent_reward.append([])
        agent_next_obs.append([])
        agent_done.append([])
        agent_mask.append([])
        agent_next_mask.append([])

    for experience in experiences:
        for i in range(self.env.config.max_num_red_agents):
            # append agent_i state: (1,g,g,ch*n_frames)
            agent_obs[i].append(experience.obss[i])

            # append agent_i action: (1,)
            agent_action[i].append(experience.actions[i])

            # append agent_i reward: (1,)
            agent_reward[i].append(experience.rewards[i])

            # append agent_i next_state: (1,g,g,ch*n_frames)
            agent_next_obs[i].append(experience.next_obss[i])

            # append agent_i done: (1,)
            agent_done[i].append(experience.dones[i])

            # append agent_i mask: (1,1,n)
            agent_mask[i].append(experience.masks[i])

            # append agent_i next_mask: (1,1,n)
            agent_next_mask[i].append(experience.next_masks[i])

        team_r.append(experience.global_r)  # append (1.1)
        team_done.append(experience.global_done)  # append (1,1)

        global_state.append(experience.global_state)  # append (1,g,g,global_ch*global_n_frames)
        next_global_state.append(experience.next_global_state)

    return agent_obs, agent_action, agent_reward, agent_next_obs, agent_done, \
           team_r, team_done, agent_mask, agent_next_mask, global_state, next_global_state


def per_agent_list2input_list(self, agent_obs, agent_action, agent_reward, agent_next_obs,
                              agent_done, team_r, team_done, agent_mask, agent_next_mask,
                              global_state, next_global_state):
    """
    per_agent_list -> input list to policy network
    :return:
        obss: [(b,g,g,ch*n_frames), ...], len=n
        actions: [(b,), ...], len=n
        rewards: [(b,), ...], len=n
        next_obss: [(b,g,g,ch*n_frames), ...], len=n
        dones: [(b,), ...], len=n, bool
        team_rs: (b,1)
        team_dones: (b,1), bool
        masks: [(b,1,n), ...], len=n, bool
        next_masks: [(b,1,n), ...], len=n, bool
        global_states: (b,g,g,global_ch*global_n_frames)
        next_global_states:  (b,g,g,global_ch*global_n_frames)
    """
    # b=self.env.config.actor_rollout_step=100
    obss = []  # [(b,g,g,ch*n_frames), ...], len=n
    actions = []  # [(b,), ...], len=n
    rewards = []  # [(b,), ...], len=n
    next_obss = []  # [(b,g,g,ch*n_frames), ...], len=n
    dones = []  # [(b,), ...], len=n, bool

    masks = []  # [(b,1,n), ...], , len=n, bool
    next_masks = []  # [(b,1,n), ...], , len=n, bool

    for i in range(self.env.config.max_num_red_agents):
        obss.append(np.concatenate(agent_obs[i], axis=0))  # append (b,15,15,6)
        actions.append(np.concatenate(agent_action[i], axis=0))  # append (b,)
        rewards.append(np.concatenate(agent_reward[i], axis=0))  # append (b,)
        next_obss.append(
            np.concatenate(agent_next_obs[i], axis=0)
        )  # append (b,15,15,6)
        dones.append(np.concatenate(agent_done[i], axis=0).astype(np.float32))  # append (b,)
        masks.append(np.concatenate(agent_mask[i], axis=0).astype(np.float32))  # append (b,1,n)
        next_masks.append(np.concatenate(agent_next_mask[i], axis=0).astype(np.float32))

    team_rs = np.concatenate(team_r, axis=0)  # (b,1)
    team_dones = np.concatenate(team_done, axis=0).astype(np.float32)  # (b,1)

    global_states = np.concatenate(global_state, axis=0)  # (b,g,g,global_ch*global_n_frames)
    next_global_states = np.concatenate(next_global_state, axis=0)

    return obss, actions, rewards, next_obss, dones, team_rs, team_dones, masks, next_masks, \
           global_states, next_global_states


def get_td_mask(config, masks):
    # masks: [(b,1,n), ...], len=n
    # td_mask: tensor of agent alive or not, bool

    td_mask = []
    for i in range(config.max_num_red_agents):
        mask = masks[i]  # mask of agent_i, (b,1,n)
        float_mask = tf.cast(mask[:, :, i], tf.float32)  # agent_i alive or not, (b,1)
        td_mask.append(float_mask)

    # list -> tensor
    td_mask = tf.concat(td_mask, axis=1)  # (b,n)

    return td_mask


def make_mask(alive_agents_ids, max_num_agents):
    """ Replace to 10_SAC_GlobalState_1 """

    mask = np.zeros(max_num_agents)  # (n,)

    for i in alive_agents_ids:
        mask[i] = 1

    mask = mask.astype(bool)  # (n,)
    mask = np.expand_dims(mask, axis=0)  # add batch_dim, (1,n)

    return mask


def buffer2per_agent_list(self):
    """
    self.buffer には、self.env.config.actor_rollout_steps = b (=16) 分の
    transition が入っている。

        transition in buffer to per_agent_list
        agent_obs = [obs list of agent_1=[(1,15,15,6), ...], len=16,
                     obs list of agent_2=[(1,15,15,6), ...], len=16,
                              ... ], len=n
        agent_action = [action list of agent_1=[(1,), ..., len=16,
                        action list of agent_2=[(1,), ..., len=16,
                              ... ], len=n
        agent_mask = [mask list of agent_1= [(1,1,15), ...], len=16,
                      mask list of agent_2= [(1,1,15), ...], len=16,
                              ... ], len=n, bool
    """

    agent_obs = []
    agent_action = []
    agent_reward = []
    agent_next_obs = []
    agent_done = []
    agent_mask = []

    for _ in range(self.env.config.max_num_red_agents):
        agent_obs.append([])
        agent_action.append([])
        agent_reward.append([])
        agent_next_obs.append([])
        agent_done.append([])
        agent_mask.append([])

    for transition in self.buffer:
        for i in range(self.env.config.max_num_red_agents):
            # append agent_i obs: (1,g,g,ch*n_frames)
            agent_obs[i].append(transition[0][i])

            # append agent_i action: (1,)
            agent_action[i].append(transition[1][i])

            # append agent_i reward: (1,)
            agent_reward[i].append(transition[2][i])

            # append agent_i next_obs: (1,g,g,ch*n_frames)
            agent_next_obs[i].append(transition[3][i])

            # append agent_i done: (1,)
            agent_done[i].append(transition[4][i])

            # append agent_i mask: (1,1,n)
            agent_mask[i].append(transition[5][i])

    return agent_obs, agent_action, agent_reward, agent_next_obs, agent_done, agent_mask


def make_padded_obs(max_num_agents, obs_shape, raw_obs):
    """ Replace to 10_SAC_GlobalState_1 """

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
    print(np.sum(next_states, axis=(2, 3, 4)))

    max_num_agents = 10
    mask = make_mask(alive_agents_ids, max_num_agents)
    print(mask)


if __name__ == '__main__':
    main()
