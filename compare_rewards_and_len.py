import os.path

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    filetype = [
        'run-.-tag-mean_episode_return of tests.csv',
        'run-.-tag-mean_episode_len of tests.csv'
    ]
    parent_dir = \
        '03_PPO/trial_gcp/'
    filelist = 'history/'

    colorlist = ['r', 'm', 'g', 'b', 'y']

    window = 10  # window of moving average

    for f, c in zip(filetype, colorlist):
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
        plt.ylabel('average return or episode length')

        label = f.replace('run-.-tag-mean_', '')
        label = label.replace('.csv', '')
        plt.plot(average_step / 1000, average_prop, linestyle='solid', color=c, alpha=0.7,
                 linewidth=1, label=label)
        plt.plot(step / 1000, prop, linestyle='solid', color=c, alpha=0.1, linewidth=1)

    # plt.yscale('log')
    plt.title(f'MA(win={window}) of returns and episode length, eval over 50 eps')
    plt.grid(which="both")
    plt.minorticks_on()
    plt.legend()

    savedir = Path(__file__).parent / (parent_dir + 'history_plots')
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    savename = 'Returns'
    plt.savefig(str(savedir) + '/' + savename + '.png', dpi=500)

    plt.show()


if __name__ == '__main__':
    main()
