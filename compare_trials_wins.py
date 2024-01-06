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
        "06_SAC_Discrete/01_Baseline/trial_1/history",
        "06_SAC_Discrete/01_Baseline/trial_3/history",
    ]

    filetypes = [
        '/run-.-tag-num_red_win.csv',
        '/run-.-tag-num_blue_win.csv',
        '/run-.-tag-num_no_contest.csv'
    ]

    legend_list_1 = ['b_16/ ', 'b_128/ ']
    legend_list_2 = ['red win', 'blue win', 'no-contest']

    colorlist = ['r', 'b', 'g']
    linelist = ['dashed', 'solid', 'dotted']

    window = 10

    for f, line, l1 in zip(filelist, linelist, legend_list_1):
        for filetype, l2, c in zip(filetypes, legend_list_2, colorlist):

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
    plt.title(f'Moving Average of Win Ratios, window={window}')
    plt.grid(which="both")
    plt.xlabel(mode + ' steps [K]')
    plt.ylabel('win ratio')
    plt.minorticks_on()
    plt.legend()
    # plt.legend(loc='upper center', bbox_to_anchor=(.5, -.1), ncol=3)

    savedir = Path(__file__).parent / 'history_plots'
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    savename = filetype.replace('/run-.-tag-', '')
    savename = savename.replace('.csv', '')
    plt.savefig(str(savedir) + '/wins' + '.png', dpi=500)

    plt.show()


if __name__ == '__main__':
    main()
