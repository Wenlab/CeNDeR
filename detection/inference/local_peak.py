# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

import os
import cv2
import sys
import numpy as np
from tqdm import tqdm
from typing import Dict, List
import matplotlib.pyplot as plt
from multiprocessing import Pool
from skimage.feature import peak_local_max

sys.path.append("../../")
plt.rcParams['figure.figsize'] = (10, 10)


def find_region_candidates_volume(volume, peak_threshold, shifts = (0, 0)):
    """

    :param volume:
    :param peak_threshold:
    :param shifts: (x, y)
    :return:
    """
    results = list()
    for i in range(volume.shape[-1]):
        slice = volume[..., i].copy()
        cv2.GaussianBlur(slice, (3, 3), 0.4, dst = slice)
        pts = peak_local_max(slice, min_distance = 1, threshold_abs = peak_threshold)  # (y, x), main
        results.extend([[int(p[1]) - shifts[0], int(p[0]) - shifts[1], i] for p in pts])  # pts shift (x, y, z)

    return results


def parse_volume_peaks_multiprocess(auto_preprocess_result: Dict, peak_threshold: int):
    """
    This version are suitable for mode 3 & 6 results
    :param auto_preprocess_result:
    :param peak_threshold:
    :return:
    """

    # name, volume, head_binary_region, rect_of_roi, peak_threshold
    params = [[name, r[0], r[2][2], r[2][3], peak_threshold] for name, r in auto_preprocess_result.items()]
    with Pool(min(len(params), os.cpu_count() // 2)) as p:
        #  [[name: [volume, peaks]], ...]
        with tqdm(p.imap_unordered(findPeaks_normalizeVolume, params), total = len(params), desc = "S.2 detection.peak_finding") as pbar:
            samples = {name: [vol, results] for name, [vol, results] in list(pbar)}
    return samples


def findPeaks_normalizeVolume(params):
    """

    :param params: volume, head_binary_region, rect_of_roi, peak_threshold
    :return:
    """

    name, volume, head_binary_region, rr, peak_threshold = params
    crv = (volume[rr[1]: rr[1] + rr[3], rr[0]: rr[0] + rr[2]]).astype(np.float32)
    crv = (crv - crv.mean()) / crv.std()
    crv = crv * head_binary_region[rr[1]: rr[1] + rr[3], rr[0]: rr[0] + rr[2], np.newaxis]
    results = find_region_candidates_volume(volume = crv, peak_threshold = peak_threshold)
    return name, [crv, results]


def pred_pt_in_bbox(pt, bbox) -> bool:
    """

    :param pt: [x, y, z]
    :param bbox: [z, xmin, ymin, xmax, ymax]
    :return:
    """

    if (pt[2] == bbox[0]) and (bbox[1] <= pt[0] < bbox[3]) and (bbox[2] <= pt[1] < bbox[4]):
        return True
    else:
        return False


def calc_tps(pts, bboxes):
    """
    volume level
    :param pts: a list of  [x, y, z]
    :param bboxes: a list of [z, xmin, ymin, xmax, ymax]
    :return:
    """

    tp = 0
    for bbox in bboxes:
        for i, pt in enumerate(pts):
            if pred_pt_in_bbox(pt, bbox):
                pts.pop(i)
                tp += 1
                break
    return tp


def calc_local_peak_score(pts: List, bboxes: List):
    """
    volume level
    :param pts: a list of  [x, y, z]
    :param bboxes: a list of [z, xmin, ymin, xmax, ymax]
    :return:
    """

    tp = calc_tps(pts.copy(), bboxes.copy())
    precision = tp / len(pts)
    recall = tp / len(bboxes)
    return precision, recall
