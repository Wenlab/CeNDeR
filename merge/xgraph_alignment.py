# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

import os
import json
import numpy as np
from tqdm import tqdm
from typing import Dict
from scipy import signal
from multiprocessing import Pool
from munkres import Munkres, make_cost_matrix

from common_utils.prints import fff


def xneuronalign_multiprocess(iou_threshold: float, span_threshold: int, volumes: Dict, volumes_2d_regions_result: Dict):
    """3D Merge multi-processing version

    :param iou_threshold:
    :param volumes:
    :param volumes_2d_regions_result:
    :return:
    """

    params = [[name, volumes[name][0], volumes_2d_regions_result[name]] for name in volumes_2d_regions_result.keys()]
    with Pool(min(os.cpu_count() // 2, len(params))) as p:
        with tqdm(p.imap_unordered(XNeuronAlign(iou_threshold, span_threshold).single_volume_align_process, params), total = len(params), desc = "S.3 merge") as pbar:
            volumes_3d_neurons_result = {name: neurons for name, neurons in list(pbar)}
    return volumes_3d_neurons_result


class XNeuronAlign(object):

    def __init__(self, iou_threshold: float = 0.05, span_threshold: int = 1) -> None:
        """3D Merging

        2D region description: [xmin, ymin, xmax, ymax, z]
        :param iou_threshold: IoU threshold of two region merge in two neighbouring slices.
        :param span_threshold: length threshold of a neuron. (should be >= 2)
        """

        super().__init__()
        self.iou_threshold = iou_threshold
        self.span_threshold = span_threshold

    def __call__(self, volumes: Dict, volumes_2d_regions_result: Dict):
        volumes_3d_neurons_result = {vol_ctn: self.single_volume_align(volumes[vol_ctn][0], volumes_2d_regions_result[vol_ctn]) for vol_ctn in tqdm(volumes_2d_regions_result.keys(), desc = " merge ")}
        return volumes_3d_neurons_result

    def single_volume_align_process(self, params):
        name, volume, bboxes = params
        volume_neurons = self.single_volume_align(volume, bboxes)
        return name, volume_neurons

    def single_volume_align(self, volume, bboxes):
        # define the maximum number of bboxes a frame as the start idx
        init_idx = int(np.argmax([len(bbox) for bbox in bboxes]))

        # calculate IOU matrixes for neighboring 2 frames
        iou_matrixes = self.calculate_iou_matrixes(bboxes, init_idx)

        # get pairs to define id for the next frame bboxes
        pairs = self.get_pairs(iou_matrixes)

        # merge 2d bboxes into 3D neuron by IOU matrixes
        volume_neurons = self.merge_neurons(bboxes, pairs, init_idx)

        # modify neurons
        volume_neurons = self.modify_neurons(volume, volume_neurons)

        return volume_neurons

    # ------------------- x.2.0 ---------------------
    @staticmethod
    def calc_iou(a: list, b: list):
        """

        :param a: [xmin, ymin, xmax, ymax, ...]
        :param b: [xmin, ymin, xmax, ymax, ...]
        :return: intersection of union between two 2d bboxes (float scalar)
        """

        inter_area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
        union_area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter_area
        iou = inter_area / (union_area + 1e-5)

        return iou

    def _calculate_iou_matrix(self, bb):
        # len(bb[0]) x len(bb[1]) matrix
        return [[self.calc_iou(b0[:-1], b1[:-1]) for b1 in bb[1]] for b0 in bb[0]]

    def calculate_iou_matrixes(self, bboxes, init_idx):
        # row: std, column: cad
        # part 1 matrixes: descending
        matrix1 = [self._calculate_iou_matrix(bboxes[i:i + 2][::-1]) for i in range(init_idx)]
        # part2 matrixes: ascending
        matrix2 = [self._calculate_iou_matrix(bboxes[i:i + 2]) for i in range(init_idx, len(bboxes) - 1)]
        return matrix1, matrix2

    # ------------------- x.2.1 ---------------------
    def _get_pairs_using_hungarian_algorithm_one_matrix(self, matrix):
        cost_matrix = make_cost_matrix(matrix, lambda iou: 1.0 - iou)
        raw_pairs = Munkres().compute(cost_matrix)  # [[y, x], ...]
        # --------------- filter pair ---------------
        pairs = list(filter(lambda p: matrix[p[0]][p[1]] >= self.iou_threshold, raw_pairs))
        return pairs

    def get_pairs(self, matrixes):
        part1, part2 = matrixes
        pairs1 = [self._get_pairs_using_hungarian_algorithm_one_matrix(m) for m in part1]
        pairs2 = [self._get_pairs_using_hungarian_algorithm_one_matrix(m) for m in part2]
        return pairs1, pairs2

    # ------------------- x.2.2 ---------------------
    def _allocate_id(self, bboxes, pairs, init_idx):

        # standard neurons (in-place)
        bboxes[init_idx] = [b + [i] for i, b in enumerate(bboxes[init_idx])]
        number_neuron = len(bboxes[init_idx]) - 1  # from 0

        # divided into 2 parts
        bboxes0, bboxes1 = bboxes[:init_idx + 1][::-1], bboxes[init_idx:]
        pairs0, pairs1 = pairs[0][::-1], pairs[1]

        # mark part0
        for idx, pair in enumerate(pairs0):
            bbox_std, bbox_cad = bboxes0[idx], bboxes0[idx + 1]
            # Allocation
            for std, cad in pair:
                bbox_cad[cad].append(bbox_std[std][-1])
            # Definition
            for bb in bbox_cad:
                if len(bb) == 5:
                    number_neuron += 1
                    bb.append(number_neuron)

        # mark part1
        for idx, pair in enumerate(pairs1):
            bbox_std, bbox_cad = bboxes1[idx], bboxes1[idx + 1]
            # Allocation
            for std, cad in pair:
                bbox_cad[cad].append(bbox_std[std][-1])
            # Definition
            for bb in bbox_cad:
                if len(bb) == 5:
                    number_neuron += 1
                    bb.append(number_neuron)

        return bboxes

    def merge_neurons(self, bboxes, pairs, init_idx):
        bboxes = [bb.tolist() for bb in bboxes]  # transfer np.ndarray into list
        bboxes = self._allocate_id(bboxes, pairs, init_idx)
        neurons = dict()
        for bb in bboxes:
            for b in bb:
                key, value = b[-1], b[:-1]
                if neurons.get(key) is None:
                    neurons[key] = [value]
                else:
                    neurons[key].append(value)
        return neurons

    # ------------------- x.2.3 ---------------------
    def _split_neuron_by_gradient(self, volume, neurons):
        for key, bb in neurons.copy().items():
            if len(bb) > 2:
                # find the local minimum and which part to allocate
                split_pairs = list()  # element 0 represents whether 1 belong the previous bb
                pixels = np.array([float(volume[b[1]: b[3], b[0]: b[2], b[-1]].mean()) for b in bb])
                local_mins = signal.argrelmin(pixels)[0].tolist()
                for lm in local_mins:
                    if self.calc_iou(bb[lm - 1], bb[lm]) >= self.calc_iou(bb[lm], bb[lm + 1]):
                        split_pairs.append([1, lm])
                    else:
                        split_pairs.append([0, lm])

                # split neuron and allocate new id
                if len(split_pairs) == 1:
                    neurons[key] = bb[:sum(split_pairs[0])]
                    neurons[max(neurons.keys()) + 1] = bb[sum(split_pairs[0]):]

                elif len(split_pairs) >= 2:
                    for idx in range(len(split_pairs)):
                        if idx == 0:
                            neurons[key] = bb[:sum(split_pairs[idx])]
                            neurons[max(neurons.keys()) + 1] = bb[sum(split_pairs[idx]): sum(split_pairs[idx + 1])]
                        elif idx == len(split_pairs) - 1:
                            neurons[max(neurons.keys()) + 1] = bb[sum(split_pairs[idx]):]
                        else:
                            neurons[max(neurons.keys()) + 1] = bb[sum(split_pairs[idx]): sum(split_pairs[idx + 1])]
        return neurons

    def _filter_neurons(self, neurons):
        """Filter some short length of neurons

        :param neurons:
        :return:
        """

        neurons = {key: values for key, values in neurons.items() if len(neurons[key]) >= self.span_threshold}
        return neurons

    def _rearrange_neurons_id(self, neurons):
        """Rearrange id, due to filter some neurons.

        :param neurons:
        :return:
        """

        new_neurons = {idx: value for idx, value in enumerate(neurons.values())}
        return new_neurons

    def modify_neurons(self, volume, neurons):
        # --------------- filter pair ---------------
        neurons = self._split_neuron_by_gradient(volume, neurons)
        # --------------- filter neuron ---------------
        neurons = self._filter_neurons(neurons) if self.span_threshold >= 2 else neurons
        neurons = self._rearrange_neurons_id(neurons)
        return neurons

    # ------------------- store ---------------------
    @staticmethod
    def save_dict_as_json(root, name, output: dict, result):
        # assert isinstance(output, dict), f'{name} must be dict type!'
        mat_name, num_volume = name[:-4], fff(int(name[-3:]) + 1, 3)
        new_dict = dict()

        ox, oy, oz = [int(x) + 1 for x in result[2][3][:2]] + [1]
        for key in sorted(output.keys()):
            new_dict[str(key + 1)] = [[i[0] + ox, i[1] + oy, i[2] + ox, i[3] + oy, i[4] + oz] for i in output[key]]

        # store
        root = os.path.join(root, mat_name)
        os.makedirs(root, exist_ok = True)
        with open(os.path.join(root, num_volume + '.json'), 'w') as f:
            json.dump(new_dict, f)
