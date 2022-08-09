# -*- coding: utf-8 -*-
# 
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com
import os
import re
import random
import numpy as np
from glob import glob
from typing import NewType, List, Tuple, Dict

DatasetIndexTuple = Tuple[int, int]


def get_stack_name(reg, path):
    group_struct = re.search(reg, path)
    return group_struct.group() if group_struct else None


def load_folders_name(reg, root):
    stack_names = [get_stack_name(reg, path) for path in glob(os.path.join(root, "*")) if get_stack_name(reg, path)]
    return stack_names


def extract_waiting_stacks(store_root, error_file_path, reg, re_infer_error, paths):
    if not os.path.exists(error_file_path):
        waiting_paths = paths
    else:
        if re_infer_error:
            with open(error_file_path, 'r') as error_file:
                waiting_paths = [path[:-1] for path in error_file.readlines()]
        else:
            with open(error_file_path, 'r') as error_file:
                error_names = [get_stack_name(reg, path) for path in error_file.readlines()]
            processed_names = load_folders_name(reg, store_root)
            assert len(set(error_names) & set(processed_names)) == 0
            filtered_names = error_names + processed_names
            waiting_paths = [path for path in paths if get_stack_name(reg, path) not in filtered_names]
            assert len(waiting_paths) >= (len(paths) - len(filtered_names))

    return sorted(waiting_paths)


def split_processing_streams(paths, max_mats_one_stream = 8):
    streams = [paths[i * max_mats_one_stream: (i + 1) * max_mats_one_stream] for i in range(int(np.ceil(len(paths) / max_mats_one_stream)))]
    return streams


def split_one_individual_dataset_name(label: Dict, train_index: DatasetIndexTuple, val_index: DatasetIndexTuple, test_index: DatasetIndexTuple, shuffle: bool = False):
    names = sorted(label.keys())
    if shuffle:
        np.random.shuffle(names)
    dataset_name = [names[i0: i1] for (i0, i1) in [train_index, val_index, test_index]]
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
