import shutil
import ray
import numpy as np
import tensorflow as tf
import json
import os
import time

from pathlib import Path

from models import MarlTransformerModel
from battlefield_strategy import BattleFieldStrategy
from utils_transformer import make_mask, make_padded_obs

from worker import Worker
from tester import Tester


def write_config(config):
    """
    Save Training conditions
    """
    config_list = {
        'max_episodes_test_play': config.max_episodes_test_play,
        'grid_size': config.grid_size,
        'offset': config.offset,

        'action_dim': config.action_dim,
        'observation_channels': config.observation_channels,
        'n_frames': config.n_frames,

        'hidden_dim': config.hidden_dim,
        'key_dim': config.key_dim,
        'num_heads': config.num_heads,
        'dropout_rate': config.dropout_rate,

        'actor_rollout_steps': config.actor_rollout_steps,
        'num_update_cycles': config.num_update_cycles,
        'batch_size': config.batch_size,
        # 'num_minibatchs': config.num_minibatchs,

        'tau': config.tau,
        'gamma': config.gamma,
        'gae_lambda': config.gae_lambda,

        'max_steps': config.max_steps,

        'learning_rate': config.learning_rate,
        'value_loss_coef': config.value_loss_coef,
        'entropy_coef': config.entropy_coef,

        'update_batch_size': config.update_batch_size,
        'opt_iter': config.opt_iter,
        'clip_range': config.clip_range,
        'clip_by_global_norm': config.clip_by_global_norm,

        'loss_coef': config.loss_coef,

        'threshold': config.threshold,
        'mul': config.mul,
        'dt': config.dt,

        'agent_types': config.agent_types,
        'agent_forces': config.agent_forces,

        'red_platoons': config.red_platoons,
        'red_companies': config.red_companies,

        'blue_platoons': config.blue_platoons,
        'blue_companies': config.blue_companies,

        'efficiencies_red': config.efficiencies_red,
        'efficiencies_blue': config.efficiencies_blue,

        'max_num_red_agents': config.max_num_red_agents,
    }

    dir_save = './result'
    if not os.path.exists(dir_save):
        os.mkdir(dir_save)

    with open(dir_save + '/training_conditions.json', 'w') as f:
        json.dump(config_list, f, indent=5)


def learn(num_workers=8, is_debug=False):
    if is_debug:
        print("Debug mode starts. May cause ray memory error.")
    else:
        print("Execution mode starts")

    ray.init(local_mode=is_debug, ignore_reinit_error=True)

    logdir = Path(__file__).parent / "log"

    summary_writer = tf.summary.create_file_writer(logdir=str(logdir))

    start = time.time()
    history = []

    # Make result dir
    resultdir = Path(__file__).parent / "result"
    if resultdir.exists():
        shutil.rmtree(resultdir)

    """ Instantiate environment """
    env = BattleFieldStrategy()
    write_config(env.config)
    action_space = env.action_space.n

    """ Instantiate & Build global policy """
    grid_size = env.config.grid_size
    ch = env.config.observation_channels
    n_frames = env.config.n_frames

    obs_shape = (grid_size, grid_size, ch * n_frames)

    max_num_agents = env.config.max_num_red_agents

    # Define alive_agents_ids & raw_obs
    alive_agents_ids = [0, 2]
    agent_obs = {}

    for i in alive_agents_ids:
        agent_id = 'red_' + str(i)
        agent_obs[agent_id] = np.ones(obs_shape)

    # Get padded_obs and mask
    padded_obs = make_padded_obs(max_num_agents, obs_shape, agent_obs)  # (1,n,g,g,ch*n_frames)

    mask = make_mask(alive_agents_ids, max_num_agents)  # (1,n)

    # Build global model & old_global_model
    global_policy = MarlTransformerModel(config=env.config)
    [policy_probs, values], scores = global_policy(padded_obs, mask, training=False)  # build
    global_policy.summary()

    old_global_policy = MarlTransformerModel(config=env.config)
    old_global_policy(padded_obs, mask, training=False)  # build

    """ Load model if necessary """
    if env.config.model_dir:
        # global_policy = tf.keras.models.load_model(env.config.model_dir)
        global_policy.load_weights(env.config.model_dir)

    """ Instantiate optimizer """
    optimizer = tf.keras.optimizers.Adam(learning_rate=env.config.learning_rate)

    """ Instantiate workers """
    workers = [Worker.remote(worker_id=i) for i in range(num_workers)]

    """ Instantiate tester """
    tester = Tester.remote()

    """ get the weights of global policy, and starts worker process """
    weights = global_policy.get_weights()
    work_in_progress = \
        [worker.rollout_and_collect_trajectory.remote(weights) for worker in workers]
    test_in_progress = tester.test_play.remote(weights)

    update_cycles = env.config.n0 + 1
    # actor_cycles = env.config.actor_cycles
    test_cycles = update_cycles

    while update_cycles <= env.config.num_update_cycles:
        """ 1. Execute worker process, and get trajectory as list """
        trajectories = ray.get(work_in_progress)

        """ 2. Starts new worker process """
        work_in_progress = \
            [worker.rollout_and_collect_trajectory.remote(weights) for worker in workers]

        """ 3. Reshape states, actions, masks, expected_returns 
                w = num_workers, b=env.config.batch_size=worker_rollout_steps
                states: (w*b, n, g, g, ch*n_frames)
                actions: (w*b, n), np.int32
                masks: (w*b, n), bool
                advantages: (w*b,n)
                expected_returns:  (w*b, n)
        """
        # Make lists
        (states, actions, masks, advantages, expected_returns) = [], [], [], [], []

        for i in range(num_workers):
            states.append(trajectories[i]["s"])  # append (b,n,g,g,ch*n_frames)
            actions.append(trajectories[i]["a"])  # append (b,n)
            masks.append(trajectories[i]["mask"])  # append (b,n)
            advantages.append(trajectories[i]["advantage"])  # append (b,n)
            expected_returns.append(trajectories[i]["R"])  # append (b,n)

        # lists -> np.array
        states = np.array(states, dtype=np.float32)  # (w,b,n,g,g,ch*n_frames)
        actions = np.array(actions, dtype=np.int32)  # (w,b,n), np.int32
        masks = np.array(masks, dtype=bool)  # (w,b,n) bool
        advantages = np.array(advantages, dtype=np.float32)  # (w,b,n)
        expected_returns = np.array(expected_returns, dtype=np.float32)  # (w,b,n)

        # reshape to batch_size=w*b
        data_size = num_workers * env.config.batch_size  # w*b
        states = states.reshape([data_size, max_num_agents, grid_size, grid_size, ch * n_frames])
        actions = actions.reshape([data_size, max_num_agents])
        masks = masks.reshape([data_size, max_num_agents])
        advantages = advantages.reshape([data_size, max_num_agents])
        expected_returns = expected_returns.reshape([data_size, max_num_agents])

        """ 
        4. Use mini_batch (update_batch_size=ub) to compute loss and grads, 
           then update trainable parameters.
           Repeat opt_iter times.
        """
        batch_size = env.config.update_batch_size
        opt_iter = env.config.opt_iter

        old_global_policy.set_weights(global_policy.get_weights())

        # Randomly select minibatch indices: (opt_iter,ub)
        indices = np.random.choice(range(states.shape[0]), size=(opt_iter, batch_size))

        sum_policy_loss = 0
        sum_value_loss = 0
        sum_entropy = 0

        for i in range(opt_iter):
            idx = indices[i]  # (ub,)

            [old_policy_probs, old_values], _ = \
                old_global_policy(states[idx], masks[idx], training=False)
            # (ub,n,action_dim), (ub,n,1)

            with tf.GradientTape() as tape:
                [policy_probs, values], _ = global_policy(states[idx], masks[idx], training=False)
                # (ub,n,action_dim), (ub,n,1)

                """ Compute log π(a|s) """
                selected_actions = tf.convert_to_tensor(actions[idx], dtype=tf.int32)  # (ub,n)
                # one_hot for dead or dummy agents' action (=-1) is zero vectors.
                selected_actions_onehot = \
                    tf.one_hot(selected_actions, depth=action_space, dtype=tf.float32)
                # (ub,n,action_dim)

                log_probs = \
                    selected_actions_onehot * tf.math.log(policy_probs + 1e-5)  # (ub,n,action_dim)
                selected_actions_log_probs = tf.reduce_sum(log_probs, axis=-1)  # (ub,n)

                """ Covert masks[idx] to tf.tensor (float32) """
                masks_ub = tf.convert_to_tensor(masks[idx], dtype=tf.float32)  # (ub,n)

                # Compute num of alive agents every batch (time step)
                num_alive_agents = tf.reduce_sum(masks_ub, axis=-1)  # (ub,)

                """ Compute lod π(a|s)_old """
                log_old_probs = \
                    selected_actions_onehot * tf.math.log(old_policy_probs + 1e-5)
                # (ub,n,action_dim)

                selected_actions_log_old_probs = tf.reduce_sum(log_old_probs, axis=-1)  # (ub,n)

                """ Compute Policy loss """
                ratio = \
                    tf.exp(selected_actions_log_probs - selected_actions_log_old_probs)  # (ub.n)

                ratio_clipped = \
                    tf.clip_by_value(ratio, 1 - env.config.clip_range, 1 + env.config.clip_range)
                # (ub,n)

                ploss_unclipped = ratio * advantages[idx]  # (ub,n)
                ploss_clipped = ratio_clipped * advantages[idx]  # (ub,n)

                ploss = tf.minimum(ploss_unclipped, ploss_clipped)  # (ub,n)
                ploss = masks_ub * ploss  # (ub,n)
                ploss = tf.reduce_sum(ploss, axis=-1)  # (ub,)
                ploss = ploss / num_alive_agents  # (ub,)

                policy_loss = tf.reduce_mean(ploss)  # (ub,)

                """ Compute Value loss """
                old_values = tf.reshape(old_values, shape=(batch_size, max_num_agents))
                # (ub,n,1)->(ub,n)
                values = tf.reshape(values, shape=(batch_size, max_num_agents))  # (ub,n,1)->(ub,n)

                vpred_clipped = old_values + \
                                tf.clip_by_value(values - old_values,
                                                 -env.config.clip_range, env.config.clip_range)
                # (ub,n)

                vloss = tf.maximum(tf.square(values - expected_returns[idx]),
                                   tf.square(vpred_clipped - expected_returns[idx]))  # (ub,n)

                vloss = masks_ub * vloss  # (ub,n)
                vloss = tf.reduce_sum(vloss, axis=-1)  # (ub,)
                vloss = vloss / num_alive_agents  # (ub,)
                value_loss = tf.reduce_mean(vloss)

                """ Compute entropy """
                entropy = - policy_probs * tf.math.log(policy_probs + 1e-5)  # (ub,n,action_dim)
                entropy = tf.reduce_mean(entropy, axis=-1)  # (ub,n)
                entropy = masks_ub * entropy  # (ub,n)
                entropy = tf.reduce_sum(entropy, axis=-1)  # (ub,)
                entropy = entropy / num_alive_agents  # (ub,)
                entropy = tf.reduce_mean(entropy)

                """ Compute total loss """
                loss = env.config.value_loss_coef * value_loss - 1 * policy_loss - \
                       1 * env.config.entropy_coef * entropy

                loss = env.config.loss_coef * loss

            grads = tape.gradient(loss, global_policy.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, env.config.clip_by_global_norm)

            # For display (tensorboard)
            sum_policy_loss += -1. * policy_loss * env.config.loss_coef
            sum_value_loss += env.config.value_loss_coef * value_loss * env.config.loss_coef
            sum_entropy += -1. * env.config.entropy_coef * entropy * env.config.loss_coef

            # 勾配を適用し、global_policyを更新
            optimizer.apply_gradients(zip(grads, global_policy.trainable_variables))

        info = {
            "policy_loss": sum_policy_loss,
            "value_loss": sum_value_loss,
            "entropy": sum_entropy
        }

        update_cycles += 1

        # get updated weights
        weights = global_policy.get_weights()

        finished_tester, _ = ray.wait([test_in_progress], timeout=0)
        if finished_tester:
            result = ray.get(finished_tester[0])

            print(f"test_cycles={test_cycles}, test_score={result['episode_rewards']}, "
                  f"episode_len={result['episode_lens']}")
            history.append((test_cycles, result['episode_rewards']))

            with summary_writer.as_default():
                tf.summary.scalar(
                    "mean_episode_return of tests", result['episode_rewards'], step=test_cycles)
                tf.summary.scalar(
                    "mean_episode_len of tests", result['episode_lens'], step=test_cycles)

                tf.summary.scalar(
                    "mean_num_alive_red_ratio", result['num_alive_reds_ratio'], step=test_cycles)
                tf.summary.scalar(
                    "mean_num_alive_red_platoon",
                    result['num_alive_red_platoon'], step=test_cycles)
                tf.summary.scalar(
                    "mean_num_alive_red_company",
                    result['num_alive_red_company'], step=test_cycles)
                tf.summary.scalar(
                    "mean_remaining_red_effective_force_ratio",
                    result['remaining_red_effective_force_ratio'], step=test_cycles)

                tf.summary.scalar(
                    "mean_num_alive_blue_ratio", result['num_alive_blues_ratio'], step=test_cycles)
                tf.summary.scalar(
                    "mean_num_alive_blue_platoon",
                    result['num_alive_blue_platoon'], step=test_cycles)
                tf.summary.scalar(
                    "mean_num_alive_blue_company",
                    result['num_alive_blue_company'], step=test_cycles)
                tf.summary.scalar(
                    "mean_remaining_blue_effective_force_ratio",
                    result['remaining_blue_effective_force_ratio'], step=test_cycles)

                tf.summary.scalar(
                    "num_red_win", result['num_red_win'], step=test_cycles)
                tf.summary.scalar(
                    "num_blue_win", result['num_blue_win'], step=test_cycles)
                tf.summary.scalar(
                    "num_draw", result['draw'], step=test_cycles)
                tf.summary.scalar(
                    "num_no_contest", result['no_contest'], step=test_cycles)

            test_cycles = update_cycles
            test_in_progress = tester.test_play.remote(weights)

        with summary_writer.as_default():
            tf.summary.scalar("policy_loss", info["policy_loss"], step=update_cycles)
            tf.summary.scalar("value_loss", info["value_loss"], step=update_cycles)
            tf.summary.scalar("entropy", info["entropy"], step=update_cycles)

        if update_cycles % 500 == 0:
            model_name = "global_policy_" + str(update_cycles)
            # global_policy.save('models/' + model_name)
            global_policy.save_weights('models/' + model_name + '/')

    else:
        ray.shutdown()

    model_name = "global_policy_" + str(update_cycles)
    # global_policy.save('models/' + model_name)
    global_policy.save_weights('models/' + model_name)


if __name__ == '__main__':
    is_debug = False  # True for debug

    learn(num_workers=6, is_debug=is_debug)  # default num_workers=6
