# -*- coding: utf-8 -*-
# 
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

import numpy as np
from typing import List


def clustering_momentum(embeds: List[np.ndarray], dis_thresh: float, alpha: float = 0.2):
    # 1. choosing a volume which possesses the maximum number of neurons
    _lens = [len(emb) for emb in embeds]
    amount_idxes = np.flip(np.argsort(_lens))  # descending
    base_idx = amount_idxes[0]

    # 2. calculating
    # v = (1-alpha) * v + alpha * dx
