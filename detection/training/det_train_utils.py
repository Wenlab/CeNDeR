# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

import os
import torch
import random
import numpy as np
from itertools import product
from typing import Dict, List
from multiprocessing import Pool
from torch.utils.data import Dataset

from detection.inference.bbox_regression_utils import calc_anchor_score, calc_regression_params
from detection.inference.dataset import make_multi_receptive_field


# ----------- Train Dataset -----------
class TrainingNeuronDataset(Dataset):
    def __init__(self, vol_peak: Dict,
                 input_size: int,
                 anchors_size: List[int],
                 patches_size: List[int],
                 labels: Dict,
                 ls: Dict,
                 fp16_mode = False,
                 iou_thresh: float = 0.20
                 ):
        """

        :param vol_peak:
        :param input_size:
        :param anchors_size:
        :param patches_size:
        :param labels:
        :param ls: label_shifts

        """

        self.fp16_mode = fp16_mode
        self.labels = {name: [[l[1] - ls[name][0], l[2] - ls[name][1], l[3] - ls[name][0], l[4] - ls[name][1], l[0]] for ll in lll.values() for l in ll] for name, lll in labels.items()}
        self.vol_peak = vol_peak
        self.input_size = input_size
        self.anchors_size = anchors_size
        self.patches_size = patches_size

        params = [[name, vol, pts, input_size, anchor_size, self.patches_size, self.labels[name], iou_thresh] for (name, (vol, pts)), anchor_size in product(self.vol_peak.items(), self.anchors_size)]
        with Pool(min(len(params), os.cpu_count() // 2)) as p:
            self.samples = [[mrf, score, deltas] for samples in p.imap_unordered(make_training_samples, params) for mrf, score, deltas in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mrf, score, deltas = self.data_aug(*self.samples[idx])
        if self.fp16_mode:
            re = torch.HalfTensor(mrf.copy()), torch.HalfTensor([score]), torch.HalfTensor(deltas)
        else:
            re = torch.FloatTensor(mrf.copy()), torch.FloatTensor([score]), torch.FloatTensor(deltas)
        return re

    def data_aug(self, mrf, score, deltas):
        """

        :param mrf:
        :param score:
        :param deltas:
        :return:
        """

        # 1. random vertical flip
        if random.choice([0, 1]):
            mrf = np.flip(mrf, axis = 2)
            deltas = [deltas[0], deltas[1], deltas[3], deltas[2]]

        # 2. random horizontal flip
        if random.choice([0, 1]):
            mrf = np.flip(mrf, axis = 1)
            deltas = [deltas[1], deltas[0], deltas[2], deltas[3]]

        # 3. random rotate
        t = np.random.choice([0, 1, 2, 3])
        if t == 1:
            mrf = np.rot90(mrf, k = 1, axes = (1, 2))
            deltas = [deltas[3], deltas[2], deltas[0], deltas[1]]
        elif t == 2:
            mrf = np.rot90(mrf, k = 2, axes = (1, 2))
            deltas = [deltas[1], deltas[0], deltas[3], deltas[2]]
        elif t == 3:
            mrf = np.rot90(mrf, k = 3, axes = (1, 2))
            deltas = [deltas[2], deltas[3], deltas[1], deltas[0]]

        return mrf, score, deltas


def choose_matched_label(anchor, labels):
    """
    left close and right open
    :param anchor: [xmin, ymin, xmax, ymax, z]
    :param labels: [xmin, ymin, xmax, ymax, z]
    :return:
    """

    scores = [calc_anchor_score(anchor, gt) for gt in labels]
    idx = int(np.argmax(scores))
    score = scores[idx]
    gt = labels[idx]

    return score, gt


def make_training_samples(params):
    """
    If one anchor has not any overlapped label, score and regression params will be calculated still.
    :param params: volume, peak_of_volume, anchor_size, patches_size, labels
    :return: [input_data, peak pt (x, y, z), regression value (hu, hl, wr, wl)]
    """

    name, vol, pts, input_size, anchor_size, patches_size, labels, score_thresh = params
    samples = list()
    upper_anchor_shift, lower_anchor_shift, left_anchor_shift, right_anchor_shift = anchor_size // 2, anchor_size - anchor_size // 2, anchor_size // 2, anchor_size - anchor_size // 2
    for x, y, z in pts:
        anchor = [x - left_anchor_shift, y - upper_anchor_shift, x + right_anchor_shift, y + lower_anchor_shift, z]
        # inputs
        anchor, mr_field = make_multi_receptive_field(pt = [x, y], anchor = anchor, image = vol[..., z], input_size = input_size, patches_size = [anchor_size] + patches_size)
        # targets
        score, gt = choose_matched_label(anchor, labels)

        if score >= score_thresh:
            deltas = calc_regression_params(peak = [x, y, z], anchor_shifts = [upper_anchor_shift, lower_anchor_shift, left_anchor_shift, right_anchor_shift], gt = gt)
        else:
            score = 0.0
            deltas = [0.0, 0.0, 0.0, 0.0]
        sample = [mr_field, score, deltas]
        samples.append(sample)

    return samples
