import os.path

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import math
import numpy as np


def main():
    mode = 'learning'
    # mode = 'pre-training'
    # mode = 'fine-tuning'

    filelist = [
        "16_MTC_SAC_SelfPlay/test_trial_0/history",
        "16_MTC_SAC_SelfPlay/test_trial_1/history",
        "17_MTC_SAC_SelfPlay_NewReward/trial/history",
    ]

    filetypes = [
        '/run-.-tag-mean_episode_team_return.csv',
        '/run-.-tag-mean_episode_return.csv',
        '/run-.-tag-mean_episode_len.csv'
    ]

    legend_list_1 = ['POMDP-1: ', 'POMDP-4: ', 'IR: ']
    legend_list_2 = ['team return', 'return', 'episode len']

    colorlist = ['b', 'r', 'g']
    linelist = ['solid', 'dotted', 'dashed']

    window = 10
    plt.figure(figsize=(16, 12))

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
            averaged_wall_time = np.array(averaged_wall_time)

            plt.plot((averaged_wall_time-averaged_wall_time[0]) / 3600, averaged_values,
                     linestyle=line, color=c, alpha=0.7, linewidth=2, label=l1 + l2)

    # plt.yscale('log')
    plt.title(f'Moving Average of returns and episode length, window={window}', fontsize="14")
    plt.grid(which="both")
    plt.xlabel(mode + ' hours', fontsize="14")
    plt.ylabel('returns / length', fontsize="14")
    plt.minorticks_on()
    plt.legend(fontsize="14")
    # plt.legend(loc='upper center', bbox_to_anchor=(.5, -.1), ncol=3)

    savedir = Path(__file__).parent / 'history_plots'
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    savename = filetype.replace('/run-.-tag-', '')
    savename = savename.replace('.csv', '')
    plt.savefig(str(savedir) + '/returns_time' + '.png', dpi=500)

    plt.show()


if __name__ == '__main__':
    main()
