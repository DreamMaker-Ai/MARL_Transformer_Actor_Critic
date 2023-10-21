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

        'max_steps': config.max_steps,

        'learning_rate': config.learning_rate,
        'value_loss_coef': config.value_loss_coef,
        'entropy_coef': config.entropy_coef,

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

    # Build global model
    global_policy = MarlTransformerModel(config=env.config)
    [policy_probs, values], scores = global_policy(padded_obs, mask, training=True)
    global_policy.summary()

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

    """ global policyの重みを取得し、それを使ってworker workersのプロセスを開始 """
    weights = global_policy.get_weights()
    work_in_progress = [worker.rollout_and_compute_grads.remote(weights) for worker in workers]
    test_in_progress = tester.test_play.remote(weights)

    update_cycles = env.config.n0 + 1
    # actor_cycles = env.config.actor_cycles
    test_cycles = update_cycles

    while update_cycles <= env.config.num_update_cycles:
        # 完了したプロセス（job）を１つ取り出し、勾配とinfoを取り出す
        finished_job, work_in_progress = ray.wait(work_in_progress, num_returns=1)
        grads, info = ray.get(finished_job)[0]
        worker_id = info["id"]

        # 勾配を適用し、global_policyを更新
        optimizer.apply_gradients(zip(grads, global_policy.trainable_variables))

        update_cycles += 1

        # 新しいworker workerのプロセスを定義して開始
        weights = global_policy.get_weights()
        work_in_progress.extend([workers[worker_id].rollout_and_compute_grads.remote(weights)])

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
            tf.summary.scalar("advantage", info["advantage"], step=update_cycles)

        if update_cycles % 5000 == 0:
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

    learn(num_workers=6, is_debug=is_debug)
