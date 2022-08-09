# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

import os
import re
import numpy as np
from glob import glob
from typing import List
from scipy.io import loadmat
from collections import OrderedDict

from src.common_utils.prints import pad_num, print_log_message, print_warning_message


def is_include(bbox: list, pts: List[np.ndarray]):
    """

    :param bbox: [z, xmin, ymin, xmax, ymax, iskey]
    :param pts: [ys, xs]
    :return:
    """
    xmin, ymin, xmax, ymax = bbox[1:5]
    ys_xmin = pts[0][pts[1] == xmin]
    ys_xmax = pts[0][pts[1] == xmax]

    if (len(ys_xmin) != 0) & (len(ys_xmax) != 0):
        return (np.min(ys_xmin) <= ymin) & (np.min(ys_xmax) <= ymin) & (np.max(ys_xmin) >= ymax) & (np.max(ys_xmax) >= ymax)
    else:
        return False


def extract_training_anno_with_mask(anno_path: str, data_root: str):
    """
    This func needs the minimum (x, y, z) from data name. And it
    :param anno_path:
    :param data_root:
    :return:
    """

    raw_labels = loadmat(anno_path)['neuron_boxes']
    _, num_frames_every_volume = raw_labels.shape
    for npy_path in sorted(list(glob(os.path.join(data_root, '*.npy')))):
        if npy_path.find('label.npy') != -1:
            continue
        labels = dict()
        ys, xs = np.nonzero(np.load(npy_path)[..., 0])
        mat_name = npy_path.split('/')[-1].split('.')[0]
        vol_idx, coord_x, coord_y, coord_z = np.array(mat_name.split('_')[-4:], dtype = np.int32)

        for frame_idx in range(num_frames_every_volume):
            raw_label_every_frame = raw_labels[vol_idx, frame_idx]
            if raw_label_every_frame.size == 0:
                pass
            else:
                raw_label_every_frame = raw_label_every_frame[0]
                for neuron in raw_label_every_frame:
                    # This name comes from .mat labels
                    neuron_id = neuron[1].squeeze() - 1
                    # Keep format consistent
                    if neuron_id.size > 1:
                        neuron_id = neuron_id[0]
                    neuron_name = neuron_id
                    # if neuron_name >= 107:
                    #     continue
                    neuron_bbox = neuron[4].squeeze()
                    # Start number of matlab is 1, but python 0.
                    neuron_bbox[0:2] -= 1
                    # Transfer absolute position at raw volume into relative position for cropped volume.
                    neuron_bbox[0:2] = neuron_bbox[0:2] - [coord_x, coord_y]  # relative coordinates
                    # (width, height) -> (xmax, ymax)
                    # neuron_bbox[2:4] += (neuron_bbox[0:2] + 1)
                    neuron_bbox[2:4] += neuron_bbox[0:2]

                    # [z, xmin, ymin, xmax, ymax, iskey]
                    neuron_label = [frame_idx - coord_z, *neuron_bbox.tolist(), neuron[0].squeeze().tolist()]
                    neuron_label = [int(item) for item in neuron_label]
                    if not is_include(neuron_label, [ys, xs]):
                        print_log_message(f"python style: {vol_idx}_{frame_idx}_{neuron_name}, matlab style: {vol_idx + 1}_{frame_idx + 1}_{int(neuron_name) + 1}")
                        continue
                    try:
                        if neuron_name not in labels.keys():
                            labels[neuron_name] = [neuron_label]
                        else:
                            labels[neuron_name].append(neuron_label)
                    except TypeError as e:
                        print_warning_message(f"{e}: vol {vol_idx + 1}, {neuron}")
        np.save(os.path.join(data_root, f'{mat_name}_label'), labels)


def extract_annos(anno_path: str, vol_idxes: List, label_reg: str, num_frame: int = 18):
    raw_labels = loadmat(anno_path)['neuron_boxes']
    stack_name = re.search(label_reg, anno_path).group()
    labels = OrderedDict()
    # assert max(vol_idxes) <= (raw_labels.shape[0] - 1)
    for vol_idx in vol_idxes:
        vol_name = f"{stack_name}_{pad_num(vol_idx, 3)}"
        vol_labels = dict()
        for slice_idx in range(num_frame):
            raw_label_slice = raw_labels[vol_idx, slice_idx]
            if raw_label_slice.size == 0:
                pass
            else:
                raw_label_slice = raw_label_slice[0]
                for neuron in raw_label_slice:

                    neuron_id = neuron[1].squeeze() - 1
                    if neuron_id.size > 1:
                        neuron_id = neuron_id[0]
                    elif neuron_id.size == 0:
                        print_warning_message(f"{stack_name} vol_{vol_idx + 1} {neuron} doesn't annotate ID, and default ID is -1")
                        neuron_id = -1
                    neuron_name = neuron_id

                    neuron_bbox = neuron[4].squeeze()
                    neuron_bbox[0:2] -= 1
                    neuron_bbox[2:4] += (neuron_bbox[0:2] + 1)

                    neuron_label = [slice_idx, *neuron_bbox.tolist(), neuron[0].squeeze().tolist()]
                    neuron_label = [int(item) for item in neuron_label]
                    if neuron_name not in vol_labels.keys():
                        vol_labels[neuron_name] = [neuron_label]
                    else:
                        vol_labels[neuron_name].append(neuron_label)

        labels[vol_name] = vol_labels
    return labels
