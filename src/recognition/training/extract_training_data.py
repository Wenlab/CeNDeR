# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

import json
import numpy as np
from glob import glob
from tqdm import tqdm
from itertools import combinations
from typing import Dict, List, Tuple
from collections import Counter, OrderedDict

from src.common_utils.prints import pad_num, print_info_message, print_error_message


def find_intersected_neuron_ids(neurons):
    """
    Calculate intersected neuron identities for given labels
    :param neurons: labels type
    :return:
    """
    ids = [set(v.keys()) for v in neurons.values()]
    intersection = ids[0].intersection(*ids[1:])
    return sorted(list(intersection))


def find_all_neuron_ids(neurons):
    """
    Calculate union neuron identities for given labels

    :param neurons: labels type
    :return:
    """
    ids = [set(v.keys()) for v in neurons.values()]
    union = ids[0].union(*ids[1:])
    return sorted(list(union))


def count_neuron_frequency(neurons):
    """
    Calculate neuron frequency for given labels sorted by id number and frequency (ensure the same frequency still keep sort by number )
    Counter usage example: https://stackoverflow.com/questions/2161752/how-to-count-the-frequency-of-the-elements-in-an-unordered-list
    :param neurons: labels type
    :return: (unique_ids, frequency)
    """
    ids = [int(i) for v in neurons.values() for i in v.keys()]
    counter = [(k, v) for k, v in sorted(Counter(ids).items(), key = lambda item: item[0])]  # sorted by id number
    counter = [(k, v) for k, v in sorted(counter, key = lambda item: item[1], reverse = True)]
    unique_ids, frequencies = [v[0] for v in counter], [v[1] for v in counter]
    return unique_ids, frequencies


def extract_dataset(neurons, given_ids, mode, id_map_rev, include_others_class):
    X, y, names = list(), list(), list()
    for neuron_name, neuron_feas in neurons.items():
        for ni, fea in neuron_feas.items():
            names.append([neuron_name, ni])
            if ni in given_ids:
                X.append(fea)
                y.append(id_map_rev[ni])
            else:
                if include_others_class:
                    X.append(fea)
                    y.append(len(given_ids))
    X, y = np.array(X, dtype = np.float32), np.array(y, dtype = np.int32)
    return X, y, names


def select_ids(neurons: dict, freq_thresh: int) -> Tuple[List, List]:
    """

    :param neurons:
    :param freq_thresh: threshold for frequency
    :return:
    """
    raw_ids, raw_freq = count_neuron_frequency(neurons)
    thresh_idx = np.where(np.array(raw_freq) >= freq_thresh)[0][-1]
    ids, freq = raw_ids[:thresh_idx], raw_freq[: thresh_idx]
    return ids, freq


def select_ids_from_multi_individuals(multi_individuals: List[dict], indial_thresh: int, freq_threshs: List[int]):
    """

    :param multi_individuals:
    :param indial_thresh: threshold for the number of individuals
    :param freq_threshs: frequency thresholds for every individual
    :return:
    """
    # 1. select ids in a individual
    indial_ids = [set(select_ids(indial, freq_thresh)[0]) for indial, freq_thresh in zip(multi_individuals, freq_threshs)]
    # 2. calculate a intersection among indial_thresh individuals by combination operation
    res = [indial_ids[t[0]].intersection(*[indial_ids[i] for i in t[1:]]) for t in combinations(list(range(len(multi_individuals))), indial_thresh)]
    # 3. union every result
    ids = res[0].union(*res[1:])
    return list(ids)


def neurons2data(neurons: Dict, dataset_names: List, given_ids: List = None, mode: tuple = (0, 1), include_others_class = False, verbose: bool = True):
    """allocating neuron id. N classes and 1 other class"""

    id_map = {i: pi for i, pi in enumerate(given_ids)}
    id_map_rev = {pi: i for i, pi in enumerate(given_ids)}
    num_ids = len(list(id_map.keys()))

    if include_others_class:
        id_map.update({len(given_ids): -1})
        num_ids += 1

    processing_ids = list(id_map.keys())
    if verbose:
        print_info_message(f"The number of processing ids: {num_ids}... | ID map: \n {id_map}")

    train_vols, val_vols, test_vols = dataset_names
    X_train, y_train, names_train = extract_dataset({key: neurons[key] for key in train_vols}, given_ids, mode, id_map_rev, include_others_class)
    X_val, y_val, names_val = extract_dataset({key: neurons[key] for key in val_vols}, given_ids, mode, id_map_rev, include_others_class)
    X_test, y_test, names_test = extract_dataset({key: neurons[key] for key in test_vols}, given_ids, mode, id_map_rev, include_others_class)

    return (X_train, y_train, names_train, train_vols), (X_val, y_val, names_val, val_vols), (X_test, y_test, names_test, test_vols), num_ids, id_map, processing_ids


def extract_preprocessing_json(root: str):
    """

    :param root:
    :return:
    """

    vols_xymin, vols_ccords = dict(), dict()
    for path in sorted(glob(root)):
        volume_name = "_".join(path[:-5].split("/")[-2:])
        with open(path, 'rb') as file:
            json_file = json.load(file)["shapes"]
        anno = dict()
        # ensure that annotation has head item
        for item in json_file:
            if item['label'] == 'head':
                anno = item
            if item['label'] == 'anterior_y':
                anterior_y = [v for v in item['points'][0]]
            if item['label'] == 'posterior_y':
                posterior_y = [v for v in item['points'][0]]
            if item['label'] == 'ventral_x':
                ventral_x = [v for v in item['points'][0]]
            if item['label'] == 'dorsal_x':
                dorsal_x = [v for v in item['points'][0]]
            if item['label'] == 'mass_of_center':
                mass_of_center = [v for v in item['points'][0]]
        if len(anno) == 0:
            continue

        # get head infos
        _, pts = pad_num(anno['group_id'], 1), np.array(anno['points'], dtype = np.int32)
        xmin, ymin = np.min(pts[:, 0]), np.min(pts[:, 1])

        # shift points of C. elegans coordinates system
        ccords = [mass_of_center, anterior_y, posterior_y, ventral_x, dorsal_x]
        ccords = [[p[0] - xmin, p[1] - ymin] for p in ccords]
        vols_xymin[volume_name] = [xmin, ymin]
        vols_ccords[volume_name] = ccords

    return vols_xymin, vols_ccords
