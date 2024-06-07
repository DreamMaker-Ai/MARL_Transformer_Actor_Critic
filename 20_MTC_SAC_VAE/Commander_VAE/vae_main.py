import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from pathlib import Path

from battlefield_for_vae_training import BattleFieldStrategy
from commander_observations_for_vae_training import commander_state_resize
from commander_vae_model import CommanderVAE


def main():
    continue_learning = False
    initial_vae_id = 0  # 0 for initial
    vae_dir = "vaes/vae_" + str(initial_vae_id) + "/"

    num_minibatches = 100
    mini_batch_size = 32

    env = BattleFieldStrategy()
    vae = CommanderVAE(env.config)
    if continue_learning:
        vae.load_weights(vae_dir)

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=env.config.learning_rate)

    # vae_loss_history = []
    # reconst_loss_history = []
    # kl_loss_history = []

    logdir = Path(__file__).parent / "log"
    summary_writer = tf.summary.create_file_writer(logdir=str(logdir))

    for batch_cycle in range(initial_vae_id, 100000):
        """ Generate batch of resized maps """
        batch_of_resized_maps = \
            generate_batch(env, mini_batch_size, num_minibatches)  # (3200,25,25,6)

        """ get minibatches & train vae """
        vae_losses = 0
        reconst_losses = 0
        kl_losses = 0

        for _ in range(num_minibatches * 5):
            """ get minibatch """
            minibatch = get_minibatch(batch_of_resized_maps, mini_batch_size)

            """ train vae """
            vae_loss, reconst_loss, kl_loss = train_vae(bce, env, minibatch, optimizer, vae)

            vae_losses += vae_loss
            reconst_losses += reconst_loss
            kl_losses += kl_loss

        # vae_loss_history.append(vae_losses)
        # reconst_loss_history.append(reconst_losses)
        # kl_loss_history.append(kl_losses)

        with summary_writer.as_default():
            tf.summary.scalar("vae_loss", vae_losses, step=batch_cycle)
            tf.summary.scalar("reconst_loss", reconst_losses, step=batch_cycle)
            tf.summary.scalar("kl_loss", kl_losses, step=batch_cycle)

        if batch_cycle % 50 == 0:
            save_dir = Path(__file__).parent / 'vaes'
            save_name = '/vae_' + str(batch_cycle) + '/'
            vae.save_weights(str(save_dir) + save_name)

        if batch_cycle % 50 == 0:
            save_dir = Path(__file__).parent / 'images'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            save_name = '/images_' + str(batch_cycle)
            plot_maps(env, vae, str(save_dir) + save_name)


def plot_maps(env, vae, save_name):
    commander_observation = env.reset()  # (g,g,6)

    resized_commander_observation = \
        commander_state_resize(commander_observation,
                               env.config.commander_grid_size)  # (25,25,6)
    resized_commander_maps = \
        np.expand_dims(resized_commander_observation, axis=0)  # (1,25,25,6)

    z_mean, _ = vae.encoder(resized_commander_maps)  # (1,128)
    reconst_maps = vae.decoder(z_mean)  # (1,25,25,6)

    map_ids = [0, 2, 4, 5]
    map_colors = ["Reds", "Blues", "Greens", "Purples"]

    for i, c in zip(map_ids, map_colors):
        plot_3_maps(commander_observation[:, :, i],
                    resized_commander_observation[:, :, i],
                    reconst_maps[0, :, :, i],
                    c,
                    save_name
                    )


def plot_3_maps(fig_1, fig_2, fig_3, fig_color, save_name):
    fig = plt.figure()
    x = 1
    y = 3

    implot1 = 1
    ax1 = fig.add_subplot(x, y, implot1)
    ax1.set_title("Original map", fontsize=10)
    plt.imshow(fig_1, cmap=fig_color)

    implot2 = 2
    ax2 = fig.add_subplot(x, y, implot2)
    ax2.set_title("Resized map", fontsize=10)
    plt.imshow(fig_2, cmap=fig_color)

    implot3 = 3
    ax3 = fig.add_subplot(x, y, implot3)
    ax3.set_title("Reconst map", fontsize=10)
    plt.imshow(fig_3, cmap=fig_color)

    plt.savefig(save_name + '_' + fig_color + '.png')
    # plt.show()

    plt.close()


def train_vae(bce, env, minibatch, optimizer, vae):
    with tf.GradientTape() as tape:
        """ reconstraction loss """
        reconst_maps, z_mean, z_log_var = vae(minibatch)

        # reconst_loss = bce(minibatch, reconst_maps)
        reconst_loss = tf.reduce_mean(
            tf.math.square(minibatch - reconst_maps),
            axis=[1, 2, 3])  # (b,)
        reconst_loss = reconst_loss * \
                       env.config.commander_grid_size * env.config.commander_grid_size

        """ KL loss """
        kl_loss = 1 + z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var)  # (b,256)
        kl_loss = -0.5 * tf.reduce_sum(kl_loss, axis=-1)  # (b,)
        # kl_loss = tf.reduce_mean(kl_loss)

        """ vae loss """
        # vae_loss = reconst_loss + kl_loss
        vae_loss = tf.reduce_mean(reconst_loss + env.config.kl_loss_coef * kl_loss)

    """ Update weights """
    variables = vae.trainable_variables
    grads = tape.gradient(vae_loss, variables)
    optimizer.apply_gradients(zip(grads, variables))

    return vae_loss.numpy(), np.mean(reconst_loss.numpy()), np.mean(kl_loss.numpy())


def get_minibatch(batch_of_resized_maps, mini_batch_size):
    data_id = np.random.choice(len(batch_of_resized_maps), mini_batch_size)
    minibatch = batch_of_resized_maps[data_id]
    return minibatch


def generate_batch(env, mini_batch_size, num_minibatches):
    batch_of_resized_maps = []
    for _ in range(num_minibatches * mini_batch_size):
        commander_observation = env.reset()  # (g,g,6)

        resized_commander_observation = \
            commander_state_resize(commander_observation,
                                   env.config.commander_grid_size)  # (25,25,6)

        batch_of_resized_maps.append(resized_commander_observation)

    batch_of_resized_maps = np.array(batch_of_resized_maps, dtype=np.float32)

    return batch_of_resized_maps


if __name__ == "__main__":
    main()
