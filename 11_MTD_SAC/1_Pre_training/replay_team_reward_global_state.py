import zlib
from dataclasses import dataclass
import numpy as np
import pickle


@dataclass
class Experience:
    obss: list
    actions: list
    rewards: list  # reference
    next_obss: list
    dones: list  # reference
    global_r: np.ndarray  # team reward
    global_done: np.ndarray  # team done
    masks: list
    next_masks: list
    global_state: np.ndarray
    next_global_state: np.ndarray


class Replay:
    def __init__(self, buffer_size, compress=True):
        self.buffer_size = buffer_size
        self.buffer = [None] * self.buffer_size

        self.count = 0  # 現在のbuffer index
        self.is_full = False  # Bufferが満杯か否か

        self.compress = compress  # 圧縮するか否か

    def add(self, transitions):
        """
        worker の経験（transitions）を追加
        transitions: [transition,...], list of transition, len=batch_size=sequence_len

        :transition: 1 experience of multi-agent
            transition = (
                self.padded_obss,       # [(1,g,g,ch*n_frames),...], len=n
                padded_actions,         # [(1,),...]
                padded_rewards,         # [(1,),...]
                next_padded_obss,       # [(1,g,g,ch*n_frames),...]
                padded_dones,           # [(1,),...], bool
                global_r,               # team_r; (1,1)
                global_done,            # team_done; (1,1), bool
                self.masks,             # [(1,1,n),...], bool
                next_masks,             # [(1,1,n),...], bool
                self.global_state,      # (1,g,g,global_ch*global_n_frames)
                next_global_state,      # (1,g,g,global_ch*global_n_frames)
                )
        """

        for transition in transitions:

            exp = Experience(*transition)

            if self.compress:
                exp = zlib.compress(pickle.dumps(exp))

            self.buffer[self.count] = exp
            self.count += 1

            if self.count == self.buffer_size:
                self.count = 0
                self.is_full = True

    def sample(self, batch_size):
        """
        batch_sizeのbatchを、優先度に従って用意
        :return:
            sampled_indices: サンプルした経験のインデクスのリスト, [int,...], len=batch_size
            experience: サンプルした経験、transitionのリスト, [transition,...], len=batch_size
        """
        if self.is_full:
            N = self.buffer_size
        else:
            N = self.count

        sampled_indices = np.random.choice(np.arange(N), size=batch_size, replace=False)

        if self.compress:
            experiences = [pickle.loads(zlib.decompress(self.buffer[idx]))
                           for idx in sampled_indices]
        else:
            experiences = [self.buffer[idx] for idx in sampled_indices]

        return experiences
