import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    with open('test_engagement/result.json', 'r', encoding='utf-8') as f:
        json_load = json.load(f)

    R0_list = json_load['R0_list']
    B0_list = json_load['B0_list']

    R0s = np.array(R0_list)
    B0s = np.array(B0_list)

    mults = B0s / R0s
    min_mult = np.floor(np.min(mults))
    max_mult = np.ceil(np.max(mults))

    winner_list = json_load['winner']

    bins = 25
    d_mult = (max_mult - min_mult) / bins

    bin_center = np.zeros(bins)
    for j in range(bins):
        bin_center[j] = min_mult + (j + 0.5) * d_mult

    red_win = np.zeros(bins)
    blue_win = np.zeros(bins)
    no_contest = np.zeros(bins)
    draw = np.zeros(bins)

    for i, mult in enumerate(mults):
        for j in range(bins):
            if (min_mult + j * d_mult <= mult) and (mult < min_mult + (j + 1) * d_mult):
                if winner_list[i] == 'red_win':
                    red_win[j] += 1
                elif winner_list[i] == 'blue_win':
                    blue_win[j] += 1
                elif winner_list[i] == 'no_contest':
                    no_contest[j] += 1
                else:
                    draw[j] += 1

    savedir = Path(__file__).parent / 'test_engagement/bar_plots'
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    plt.bar(bin_center, red_win, color='r', width=d_mult, label='red win')
    plt.bar(bin_center, blue_win, bottom=red_win, color='b', width=d_mult, label='blue win')
    plt.bar(bin_center, no_contest, bottom=red_win + blue_win, color='y', width=d_mult,
            label='no-contest')
    plt.title('Num of wins vs B0/R0 over 1000 test episodes')
    plt.xlabel('B0/R0')
    plt.ylabel('Num of wins')
    plt.legend()
    plt.grid()

    savename = 'Stacked Graph'
    plt.savefig(str(savedir) + '/' + savename + '.png', dpi=300)

    plt.show()

    plt.bar(bin_center, blue_win, align='edge', color='b', width=d_mult / 2, label='blue win')
    plt.bar(bin_center, no_contest, align='center', color='y', width=d_mult / 2, label='no-contest')
    plt.bar(bin_center, red_win, align='edge', color='r', width=d_mult / 2, label='red win')
    plt.title('Num of wins vs B0/R0 over 1000 test episodes')
    plt.xlabel('B0/R0')
    plt.ylabel('Num of wins')
    plt.legend()
    plt.grid()

    savename = 'Bar Graph'
    plt.savefig(str(savedir) + '/' + savename + '.png', dpi=300)

    plt.show()

    red_win = winner_list.count('red_win')
    blue_win = winner_list.count('blue_win')
    no_contest = winner_list.count('no_contest')

    print(f'red_win: {red_win}')
    print(f'blue_win: {blue_win}')
    print(f'no_contest: {no_contest}')


if __name__ == '__main__':
    main()
