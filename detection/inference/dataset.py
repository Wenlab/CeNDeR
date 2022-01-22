# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

import os
import cv2
import torch
import numpy as np
from typing import Dict, List
from itertools import product
from multiprocessing import Pool
from torch.utils.data import Dataset


# ----------- Inference Dataset -----------
class InferringNeuronDataset(Dataset):
    def __init__(self, vol_peak: Dict,
                 input_size: int,
                 anchors_size: List[int],
                 patches_size: List[int],
                 num_workers: int,
                 fp16_mode: bool = False,
                 ):
        """

        :param vol_peak:
        :param input_size:
        :param anchors_size:
        :param patches_size:
        """

        self.fp16_mode = fp16_mode
        self.vol_peak = vol_peak
        self.input_size = input_size
        self.anchors_size = anchors_size
        self.patches_size = patches_size

        params = [[name, vol, res, input_size, anchor_size, self.patches_size] for (name, (vol, res)), anchor_size in product(self.vol_peak.items(), self.anchors_size)]

        with Pool(min(num_workers, os.cpu_count() // 2)) as p:
            # name: [[volume, peaks], ...]
            self.samples = [[name, peak, shift, mrf] for name, samples in p.imap_unordered(make_inferring_samples, params) for peak, shift, mrf in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """

        :param idx:
        :return: name, (xmin, ymin, xmax, ymax, z), mrf. the first two are to recover original position
        """

        name, peak, shift, mrf = self.samples[idx]
        if self.fp16_mode:
            re = name, torch.HalfTensor(peak), torch.HalfTensor(shift), torch.HalfTensor(mrf)
        else:
            re = name, torch.IntTensor(peak), torch.IntTensor(shift), torch.FloatTensor(mrf)

        return re


def make_inferring_samples(params):
    name, vol, pts, input_size, anchor_size, patches_size = params
    samples = list()
    upper_anchor_shift, lower_anchor_shift = anchor_size // 2, anchor_size - anchor_size // 2
    left_anchor_shift, right_anchor_shift = anchor_size // 2, anchor_size - anchor_size // 2
    for x, y, z in pts:
        anchor = [x - left_anchor_shift,
                  y - upper_anchor_shift,
                  x + right_anchor_shift,
                  y + lower_anchor_shift,
                  z]
        # inputs
        anchor, mr_field = make_multi_receptive_field(pt = [x, y],
                                                      anchor = anchor,
                                                      image = vol[..., z],
                                                      input_size = input_size,
                                                      patches_size = [anchor_size] + patches_size)
        peak = [x, y, z]
        shift = [y - anchor[1], anchor[3] - y, x - anchor[0], anchor[2] - x]  # upper_y, lower_y, left_x, right_x
        samples.append([peak, shift, mr_field])

    return name, samples


# ----------- Common func for training and inference -----------
def make_multi_receptive_field(pt: [int, int],
                               anchor: [int, int, int, int],
                               image: np.ndarray,
                               input_size: int,
                               patches_size: List
                               ) -> [List, np.ndarray]:
    """

    :param pt:
    :param anchor:
    :param image:
    :param input_size:
    :param patches_size:
    :return: (anchor, mr_field).
              anchor are modified due to out of range in the image ;
              mr_field
    """

    mr_field = np.zeros((len(patches_size), input_size, input_size), dtype = np.float32)
    ctr_h, ctr_w = input_size // 2, input_size // 2  # y, x
    height, width = image.shape
    # anchor
    # for out of range situation
    delta_x1 = min(0, anchor[0] - 0)
    delta_y1 = min(0, anchor[1] - 0)
    delta_x2 = min(0, width - anchor[2])
    delta_y2 = min(0, height - anchor[3])

    # (x1, y1, x2, y2, z)
    anchor = [anchor[0] - delta_x1, anchor[1] - delta_y1, anchor[2] + delta_x2, anchor[3] + delta_y2, anchor[-1]]

    mr_field[0, ctr_h - patches_size[0] // 2 - delta_y1: ctr_h + patches_size[0] - patches_size[0] // 2 + delta_y2,
    ctr_w - patches_size[0] // 2 - delta_x1: ctr_w + patches_size[0] - patches_size[0] // 2 + delta_x2
    ] = image[anchor[1]: anchor[3], anchor[0]: anchor[2]]
    # patches
    for i in range(1, len(patches_size)):
        # extract patch from raw data
        delta_x1 = min(0, (pt[0] - patches_size[i] // 2) - 0)
        delta_y1 = min(0, (pt[1] - patches_size[i] // 2) - 0)
        delta_x2 = min(0, width - (pt[0] + patches_size[i] - patches_size[i] // 2))
        delta_y2 = min(0, height - (pt[1] + patches_size[i] - patches_size[i] // 2))
        patch = image[
                pt[1] - patches_size[i] // 2 - delta_y1:pt[1] + patches_size[i] - patches_size[i] // 2 + delta_y2,
                pt[0] - patches_size[i] // 2 - delta_x1:pt[0] + patches_size[i] - patches_size[i] // 2 + delta_x2]

        patch = np.pad(patch, ((-delta_y1, -delta_y2), (-delta_x1, -delta_x2)), mode = 'constant', constant_values = 0)
        # assert patch.shape == (patches_size[i], patches_size[i]), "Wrong!"

        # load
        if patches_size[i] < input_size:
            # padding load
            mr_field[i,
            ctr_h - patches_size[i] // 2: ctr_h + patches_size[i] - patches_size[i] // 2,
            ctr_w - patches_size[i] // 2: ctr_w + patches_size[i] - patches_size[i] // 2] = patch
        elif patches_size[i] > input_size:
            # resize load
            mr_field[i] = cv2.resize(patch.copy(), (input_size, input_size), interpolation = cv2.INTER_LINEAR)
        else:
            mr_field[i] = patch  # if patch_size == input_size, directly load

    return anchor, mr_field
