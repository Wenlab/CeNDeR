# -*- coding: utf-8 -*-
# 
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

import random
import numpy as np
from typing import NewType, List, Tuple, Dict

DatasetIndexTuple = Tuple[int, int]


def split_processing_streams(paths, max_mats_one_stream = 8):
    streams = [paths[i * max_mats_one_stream: (i + 1) * max_mats_one_stream] for i in range(int(np.ceil(len(paths) / max_mats_one_stream)))]
    return streams


def split_one_individual_dataset_name(label: Dict, train_index: DatasetIndexTuple, val_index: DatasetIndexTuple, test_index: DatasetIndexTuple, shuffle: bool = False):
    names = sorted(label.keys())
    dataset_name = [names[i0: i1] for (i0, i1) in [train_index, val_index, test_index]]
    dataset_name = [random.sample(p, len(p)) for p in dataset_name] if shuffle else dataset_name
    return dataset_name


def split_multi_individuals_datasets(labels: List[Dict], indexes: List[List[DatasetIndexTuple]], shuffle_type: int = 0):
    """

    :param labels:
    :param indexes:
    :param shuffle_type: 0 - no shuffle, 1 - shuffle within a individual, 2 - shuffle all
    :return:
    """
    merged_label = {k: v for label in labels for k, v in label.items()}

    train_names, val_names, test_names = list(), list(), list()
    for label, index_tuple in zip(labels, indexes):
        train_name, val_name, test_name = split_one_individual_dataset_name(label, *index_tuple, shuffle = (shuffle_type == 1))
        train_names.extend(train_name)
        val_names.extend(val_name)
        test_names.extend(test_name)
    if shuffle_type == 2:
        train_len, val_len, test_len = sum([idx[0][1] - idx[0][0] for idx in indexes]), sum([idx[1][1] - idx[1][0] for idx in indexes]), sum([idx[2][1] - idx[2][0] for idx in indexes])
        names = random.sample(train_names + val_names + test_names, train_len + val_len + test_len)
        train_names, val_names, test_names = names[: train_len], names[train_len: train_len + val_len], names[train_len + val_len:]
    train_names, val_names, test_names = sorted(train_names), sorted(val_names), sorted(test_names)

    selected_label = {k: merged_label[k] for k in sorted(train_names + val_names + test_names)}
    return selected_label, (train_names, val_names, test_names)
