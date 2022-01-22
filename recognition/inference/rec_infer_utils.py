# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

import os
import time
import numpy as np
from typing import Dict, List

from merge.xgraph_alignment import XNeuronAlign
from common_utils.prints import print_info_message


def proposal_prediction(merge_results: Dict, raw_results: Dict, id_map: Dict, others_class_start_id: int = 1000):
    outputs = dict()
    for vol_name, (merged_ids, probs) in raw_results.items():
        # output = maximum_score_algorithm(merged_ids, probs, num_neuron = len(id_map), others_class_start_id = others_class_start_id)
        # outputs[vol_name] = {id_map.get(pred_id, pred_id): [merge_results[vol_name][m] for m in merged_ids] for pred_id, merged_ids in output.items()}
        output = high_score_filter_algorithm(merged_ids, probs, num_neuron = len(id_map), others_class_start_id = others_class_start_id)
        outputs[vol_name] = {id_map.get(pred_id, pred_id): merge_results[vol_name][merged_id] for pred_id, merged_id in output.items()}
    return outputs


def maximum_score_algorithm(merged_ids: List, probs: np.ndarray, num_neuron: int, others_class_start_id: int = 1000):
    """With Other class version

    :param merged_ids:
    :param probs:
    :param num_neuron:
    :param others_class_start_id:
    :return: a map dict of abstract id and the id allocating by XNeuroAlignment Algorithm.
    """
    pred_dict = dict()
    others_info = list()
    for p_i, m_i in zip(np.argmax(probs, axis = 1), merged_ids):
        if p_i != (num_neuron - 1):
            pred_dict[p_i] = pred_dict.get(p_i, list()) + [m_i]
        else:
            others_info.append(m_i)
    pred_dict.update({i + others_class_start_id: [m_i] for i, m_i in enumerate(others_info)})
    return pred_dict


def high_score_filter_algorithm(merged_id_list: List, score_matrix: np.ndarray, num_neuron: int, others_class_start_id: int = 1000):
    """With Other class version

    :param score_matrix:
    :param merged_id_list:
    :param num_neuron:
    :param others_class_start_id:
    :return: a map dict of abstract id and the id allocating by XNeuroAlignment Algorithm.
    """

    pred_dict = dict()
    has_been_done = list()
    score_array = score_matrix.flatten()
    score_arg_array = np.argsort(-score_array)
    # filter top-k neuron into out result
    for score_arg in score_arg_array[:min(score_matrix.shape[0], num_neuron) * num_neuron]:
        row, col = divmod(score_arg, num_neuron)
        m_id = merged_id_list[row]
        if (pred_dict.get(col) == None) and (m_id not in has_been_done) and (col != (num_neuron - 1)):
            pred_dict[col] = m_id
            has_been_done.append(m_id)
    # and the others are given reserved id
    reserved_id = list(set(merged_id_list) - set(has_been_done))
    for index, r_id in zip(range(others_class_start_id, others_class_start_id + len(reserved_id), 1), reserved_id):
        pred_dict[index] = r_id
    return pred_dict


def store_result_as_json(root, outputs, results):
    """

    :param root:
    :param outputs: recognition outputs
    :param results:  pre-processing results
    :return:
    """

    [XNeuronAlign.save_dict_as_json(root, name, outputs[name], result) for name, result in results.items()]
    print_info_message(f"Results have been saved in {root} !")
