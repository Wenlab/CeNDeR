# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com
import os.path

import numpy as np
from typing import Dict, List

from scipy.optimize import linear_sum_assignment
from src.merge.xgraph_alignment import XNeuronAlign
from src.common_utils.prints import print_info_message


def proposal_prediction(merge_results: Dict, raw_results: Dict, template_volume_path: str, others_class_start_id: int = 1000):
    if os.path.isfile(template_volume_path):
        template_volume = np.load(template_volume_path)
    else:
        tep_idx = np.argmax([len(res[0]) for res in raw_results.values()])
        tep_name = list(raw_results.keys())[tep_idx]
        template_volume = raw_results[tep_name][1]
        print_info_message(f"Template volume name: {tep_name}, num of neurons: {template_volume.shape[0]}")
        np.save(template_volume_path, template_volume)
    outputs = dict()
    for vol_name, (merged_ids, embeds) in raw_results.items():
        output = hungarian_matching(merged_ids, embeds, template_volume, others_class_start_id)
        outputs[vol_name] = {pred_id: merge_results[vol_name][merged_id] for pred_id, merged_id in output.items()}
    return outputs


def hungarian_matching(merged_ids: List, embeds: np.ndarray, template_volume: int, others_class_start_id: int = 1000):
    cost_matrix = 1 - np.matmul(embeds, np.transpose(template_volume))
    preds = linear_sum_assignment(cost_matrix)
    pred_dict = {p2: merged_ids[p1] for p1, p2 in zip(*preds)}
    if embeds.shape[0] > len(preds[0]):
        reserved_id = list(set(range(len(preds[0]))) - set(preds[0]))
        for p2, p1 in zip(range(others_class_start_id, others_class_start_id + len(reserved_id), 1), reserved_id):
            pred_dict[p2] = merged_ids[p1]
    return pred_dict


def store_result_as_json(root, outputs, preprc_results):
    """

    :param root:
    :param outputs: recognition outputs
    :param preprc_results:  pre-processing results
    :return:
    """

    [XNeuronAlign.save_dict_as_json(root, name, outputs[name], result) for name, result in preprc_results.items()]
    print_info_message(f"Results have been saved in {root} !")
