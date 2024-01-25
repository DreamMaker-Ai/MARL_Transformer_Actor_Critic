import zlib
from dataclasses import dataclass
import numpy as np
import pickle


@dataclass
class Experience:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    global_r: np.ndarray
    global_done: np.ndarray
    masks: np.ndarray
    next_masks: np.ndarray
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
        actorの経験（transitions）を追加

        transitions=[transition,...], list of transition
            transition =
                (
                    self.padded_states,  # (1,n,g,g,ch*n_frames)
                    padded_actions,  # (1,n)
                    padded_rewards,  # (1,n)
                    next_padded_states,  # (1,n,g,g,ch*n_frames)
                    padded_dones,  # (1,n), bool
                    global_r,  # (1,1)
                    global_done,  # (1,1), bool
                    self.mask,  # (1,n), bool
                    next_mask,  # (1,n), bool
                    global_state,  # (1,g,g,global_ch*global_n_frames)
                    next_global_state,  # (1,g,g,global_ch*global_n_frames)
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
