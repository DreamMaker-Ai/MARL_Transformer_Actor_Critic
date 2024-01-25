import os.path

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    filetype = [
        'run-.-tag-mean_num_alive_red_ratio.csv',
        'run-.-tag-mean_num_alive_blue_ratio.csv',
        'run-.-tag-mean_remaining_red_effective_force_ratio.csv',
        'run-.-tag-mean_remaining_blue_effective_force_ratio.csv',
    ]
    parent_dir = \
        '10_SAC_GlobalState_1/1_Pre_training/trial/'
    filelist = 'history/'

    colorlist = ['r', 'b', 'r', 'b']
    linelist = ['solid', 'solid', 'dotted', 'dotted']

    window = 10  # window of moving average

    for f, c, l in zip(filetype, colorlist, linelist):
        ff = parent_dir + filelist + f
        csv_path = Path(__file__).parent / ff

        csv_df = pd.read_csv(csv_path)

        wall_time = csv_df[csv_df.columns[0]]
        step = csv_df[csv_df.columns[1]]
        prop = csv_df[csv_df.columns[2]]

        # Compute rolling average
        average_step = step.rolling(window=window).mean()
        average_prop = prop.rolling(window=window).mean()

        plt.xlabel('learning steps [K]')
        plt.ylabel('average survive agents or remaining forces ratio')

        label = f.replace('run-.-tag-mean_', '')
        label = label.replace('.csv', '')
        plt.plot(average_step / 1000, average_prop, linestyle=l, color=c, alpha=1.0,
                 linewidth=1, label=label)
        plt.plot(step / 1000, prop, linestyle=l, color=c, alpha=0.3, linewidth=1)

    # plt.yscale('log')
    plt.title(f'MA(win={window}) of survive agents & remaining forces, eval over 50 eps')
    plt.grid(which="both")
    # plt.ylim([-0.1, 0.95])
    plt.minorticks_on()
    plt.legend()
    # plt.legend(loc='upper center', bbox_to_anchor=(.5, -.05), ncol=4)

    savedir = Path(__file__).parent / (parent_dir + 'history_plots')
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    savename = 'Alive agents and Forces'
    plt.savefig(str(savedir) + '/' + savename + '.png', dpi=500)

    plt.show()


if __name__ == '__main__':
    main()
