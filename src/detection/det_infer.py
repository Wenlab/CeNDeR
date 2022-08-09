# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict
from multiprocessing.pool import Pool
from torch.utils.data import DataLoader

from src.detection.inference.dataset import InferringNeuronDataset
from src.detection.inference.network import MFDetectNetworkModule41
from src.detection.inference.nms import non_maximum_suppression_volume_func
from src.detection.inference.bbox_regression_utils import calc_predicted_bbox, filter_bboxes_by_area


def infer(network, vol_peaks, args):
    candidate_names, candidate_scores, candidate_bboxes = list(), list(), list()
    for vps in tqdm(np.array_split(list(vol_peaks.keys()), np.ceil(len(vol_peaks) / 30)), desc = "S.2 detection.ANN"):
        # ----------- Dataloader -----------
        dataset = InferringNeuronDataset(vol_peak = {n: vol_peaks[n] for n in vps},
                                         input_size = args.det_input_size,
                                         anchors_size = args.det_anchors_size,
                                         patches_size = args.det_patches_size,
                                         num_workers = args.det_num_workers,
                                         fp16_mode = args.det_fp16)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = args.det_batch_size, drop_last = False, shuffle = True, pin_memory = False, num_workers = args.det_num_workers)
        # ----------- Calculation -----------
        with torch.no_grad():
            for names, peaks, shifts, mrfs in dataloader:
                pred_score, pred_delta = network(mrfs.cuda())
                pred_score, pred_delta = torch.sigmoid(pred_score).cpu(), pred_delta.cpu()
                pos_idxes = torch.nonzero(pred_score > args.det_label_iou_threshold)[:, 0]
                bboxes = [calc_predicted_bbox(p.tolist(), s.tolist(), d.tolist()) for p, s, d in zip(peaks[pos_idxes], shifts[pos_idxes], pred_delta[pos_idxes])]

                b_idxes = [i for i, b in enumerate(bboxes) if filter_bboxes_by_area(b)]
                pos_idxes = pos_idxes[b_idxes]

                candidate_names.extend(names[p] for p in pos_idxes)
                candidate_scores.extend(float(s) for s in pred_score[pos_idxes])
                candidate_bboxes.extend(bboxes[i] for i in b_idxes)

    return candidate_names, candidate_scores, candidate_bboxes


def nms_multiprocess(scores: Dict, bboxes: Dict, iou_thresh: float, number_threshold: int):
    params = [[name, bboxes[name], scores[name], iou_thresh, number_threshold] for name in scores.keys()]

    with Pool(min(os.cpu_count() // 2, len(params))) as p:
        with tqdm(p.imap_unordered(non_maximum_suppression_volume_func, params), total = len(params), desc = "S.2 detection.NMS") as pbar:
            outs = list(pbar)
    result = {out[0]: out[1] for out in outs if out != None}

    return result


def det_infer_main(args, vol_peaks):
    # ----------- Network -----------
    network = MFDetectNetworkModule41(num_channels = len(args.det_patches_size) + 1).cuda()
    network.load_state_dict(torch.load(args.det_model_load_path, map_location = 'cuda:0')['network'])
    network = network.half() if args.det_fp16 else network
    network.eval()

    # ----------- Main Procedure -----------
    cn, cs, cb = infer(network, vol_peaks, args)

    # list -> dict
    candidate_scores, candidate_bboxes = dict(), dict()
    [candidate_scores.update({n: candidate_scores.get(n, []) + [s]}) for n, s in zip(cn, cs)]
    [candidate_bboxes.update({n: candidate_bboxes.get(n, []) + [b]}) for n, b in zip(cn, cb)]
    results = nms_multiprocess(candidate_scores, candidate_bboxes, args.det_nms_iou_threshold, args.det_number_threshold)

    return results
