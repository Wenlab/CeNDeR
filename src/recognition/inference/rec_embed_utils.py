# -*- coding: utf-8 -*-
# 
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com


import os
import time
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List
from torch.utils.data import DataLoader

from src.merge.xgraph_alignment import XNeuronAlign
from src.common_utils.prints import print_info_message
from src.recognition.inference.dataset import RecFeatureDataset
from src.common_utils.metric.rec import top_k_accuracy_score, top_1_accuracy_score, top_k_accuracy


# train
# label, prediction, embedding vector
def store_neuronal_embedding(save_root: str, time_stamp: str, sum_accuracy: float, vol_results: Dict):
    path = os.path.join(save_root, time_stamp)
    os.makedirs(path, exist_ok = True)
    for name, (label, embed, pred, top1_acc, top2_acc, top3_acc) in vol_results.items():
        # res = np.concatenate((label[:, np.newaxis], pred[:, np.newaxis], embed), axis = 1)
        res = [sum_accuracy, top1_acc, top2_acc, top3_acc, label, pred, embed]
        np.save(os.path.join(path, name), res, allow_pickle = True)


def extract_embedding_wlabel(Xs, ys, sample_names, test_vols, num_ids, model, batch_size, is_fp16):
    """

    :param Xs:
    :param ys:
    :param sample_names:
    :param test_vols:
    :param num_ids:
    :param model:
    :param batch_size:
    :param is_fp16:
    :return: a top-1 accuracy of all volumes and every volume result [label, embedding vector, prediction, accuracy]
    """
    dataset = RecFeatureDataset(Xs, ys, sample_names, is_train = False, is_fp16 = is_fp16)
    dataloader = DataLoader(dataset, batch_size = batch_size, drop_last = False, shuffle = True, pin_memory = False, num_workers = 1)

    names, labels = list(), list()
    preds = torch.zeros((len(dataset), num_ids), dtype = torch.float16 if is_fp16 else torch.float32).cuda()
    embeds = torch.zeros((len(dataset), model.output_dim), dtype = torch.float16 if is_fp16 else torch.float32).cuda()
    model.eval()
    with torch.no_grad():
        for idx, (ns, feas, ids) in enumerate(dataloader):
            embeds[idx * batch_size: idx * batch_size + len(ids)], preds[idx * batch_size: idx * batch_size + len(ids)] = model(feas.cuda(), mode = 3)
            labels.extend(ids.tolist())
            names.extend([vol_name for vol_name in ns[0]])
    embeds, preds, labels = np.array(embeds.cpu()), np.array(preds.cpu()), np.array(labels)
    # load info
    vol_info = {vol_name: [[], [], []] for vol_name in sorted(test_vols)}
    for name, label, embedding, pred_vector in zip(names, labels, embeds, preds):
        vol_info[name][0].extend(label)
        vol_info[name][1].append(embedding)
        vol_info[name][2].append(pred_vector)

    # store results
    sum_accuracy = top_1_accuracy_score(labels, preds)
    vol_results = {name: [np.array(label, dtype = np.int), np.array(embeds, dtype = np.float32), np.argmax(np.array(pred), axis = 1).astype(np.int),
                          top_1_accuracy_score(np.array(label), np.array(pred)), top_k_accuracy(np.array(label), np.array(pred), k = 2), top_k_accuracy(np.array(label), np.array(pred), k = 3)]
                   for name, (label, embeds, pred) in vol_info.items()}

    return sum_accuracy, vol_results
