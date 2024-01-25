import os.path

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import math
import numpy as np


def main():
    # mode = 'learning'
    mode = 'pre-training'
    # mode = 'fine-tuning'

    filelist = [
        "08_SAC_TeamReward_1/1_Pre-training/trial_1/history",
        "10_SAC_GlobalState_1/1_Pre_training/trial/history",
    ]

    filetypes = [
        '/run-.-tag-mean_episode_team_return.csv',
        '/run-.-tag-mean_episode_return.csv',
        '/run-.-tag-mean_episode_len.csv'
    ]

    legend_list_1 = ['w/o global_state ', 'w/ global_state ']
    legend_list_2 = ['team return', 'return', 'episode len']

    colorlist = ['b', 'r', 'g']
    linelist = ['solid', 'dotted', 'dashed']

    window = 10

    for f, c, l1 in zip(filelist, colorlist, legend_list_1):
        for filetype, l2, line in zip(filetypes, legend_list_2, linelist):

            ff = f + filetype
            csv_path = Path(__file__).parent / ff

            csv_df = pd.read_csv(csv_path)

            wall_time = csv_df[csv_df.columns[0]]
            step = csv_df[csv_df.columns[1]]
            values = csv_df[csv_df.columns[2]]

            averaged_wall_time = []
            averaged_step = []
            averaged_values = []

            for idx in range(len(values) - window + 1):
                averaged_wall_time.append(
                    np.mean(wall_time[idx:idx + window])
                )

                averaged_step.append(
                    np.mean(step[idx:idx + window])
                )

                averaged_values.append(
                    np.mean(values[idx:idx + window])
                )

            averaged_step = np.array(averaged_step)
            averaged_values = np.array(averaged_values)

            plt.plot(averaged_step / 1000, averaged_values, linestyle=line, color=c,
                     alpha=0.7, linewidth=1, label=l1 + l2)

    # plt.yscale('log')
    plt.title(f'Moving Average of returns and episode length, window={window}')
    plt.grid(which="both")
    plt.xlabel(mode + ' steps [K]')
    plt.ylabel('returns / length')
    plt.minorticks_on()
    plt.legend()
    # plt.legend(loc='upper center', bbox_to_anchor=(.5, -.1), ncol=3)

    savedir = Path(__file__).parent / 'history_plots'
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    savename = filetype.replace('/run-.-tag-', '')
    savename = savename.replace('.csv', '')
    plt.savefig(str(savedir) + '/returns' + '.png', dpi=500)

    plt.show()


if __name__ == '__main__':
    main()
