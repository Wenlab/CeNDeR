# -*- coding: utf-8 -*-
# 
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com
import os
import sys
import torch
import random
import argparse
import numpy as np
from glob import glob
from torch.nn import functional as F
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment

sys.path.append("/home/cbmi/wyx/CenDer_PLOS_CompBio")
# from src.common_utils.metric.rec import top_1_accuracy_score_torch
from src.common_utils.prints import print_info_message, print_log_message
from src.benchmarks.configs import fDNC as config_fdnc
from src.benchmarks.configs import CeNDeR as config_cender
from src.benchmarks.datasets.fDNC import Dataset_fDNC, collate_fn_volumes
from src.recognition.inference.network import RecFormer_Test


def find_match(label2, label1):
    """https://github.com/XinweiYu/fDNC_Neuron_ID/blob/master/src/model.py#L11-L29"""

    # label 1 and label 2 is the label for neurons in two worms.
    if len(label1) == 0 or len(label2) == 0:
        return []
    pt1_dict = dict()
    # build a dict of neurons in worm1
    label1_indices = np.where(label1 >= 0)[0]
    for idx1 in label1_indices:
        pt1_dict[label1[idx1]] = idx1
    # search label2 in label 1 dict
    match = list()
    unlabel = list()
    label2_indices = np.where(label2 >= -1)[0]
    for idx2 in label2_indices:
        if label2[idx2] in pt1_dict:
            match.append([idx2, pt1_dict[label2[idx2]]])
        else:
            unlabel.append(idx2)
    return np.array(match), np.array(unlabel)


def calc_volume_accuracy(ref, vol, match_dict: dict):
    n_gt, n_tp = 0, 0
    for r_idx, r in enumerate(ref):
        if r in match_dict:
            n_gt += 1
            if match_dict[r] == vol[r_idx]:
                n_tp += 1
    acc = n_tp / n_gt
    return acc, n_tp, n_gt


def evaluate_benchmark(dataloader, model, refer_idx: int = 0, verbose: bool = False):
    embeds_list, lens_list, ids_list, outliers_list = list(), list(), list(), list()
    with torch.inference_mode():
        for idx, (feas, ids, lens) in enumerate(dataloader):
            embeds, outliers = model(feas[:, :, :3].cuda() if model.input_dim == 3 else feas.cuda(), lens.cuda())
            outliers_list.append(outliers.detach().cpu())
            embeds_list.append(embeds.detach().cpu())
            lens_list.append(lens)
            ids_list.append(ids)

    num_gts_argmin, num_tps_argmin = 0, 0
    num_gts_hungarian, num_tps_hungarian = 0, 0
    for batch_idx, (outliers, embeds, lens, ids) in enumerate(zip(outliers_list, embeds_list, lens_list, ids_list)):
        ref_len = lens[refer_idx]
        ref_embed = embeds[refer_idx][:ref_len]
        ref_id = ids[refer_idx][:ref_len].numpy()
        for i, (outlier, embed, _len, id) in enumerate(zip(outliers, embeds, lens, ids)):
            if i != refer_idx:
                outlier = outlier[:_len]
                embed = embed[:_len]
                id = id.numpy()[:_len]
                match_dict = {gt_m[0]: gt_m[1] for gt_m in find_match(id, ref_id)[0]}
                cost_matrix = (- F.log_softmax(torch.cat((torch.mm(embed, ref_embed.transpose(1, 0)), outlier), dim = 1), dim = 1)).numpy()

                argmin_pred = np.argmin(cost_matrix, axis = 1)  # 1-dim np.ndarray
                hungarian_pred = linear_sum_assignment(cost_matrix)  # 2-dim np.ndarray

                hug_acc, hug_num_tp, hug_num_gt = calc_volume_accuracy(hungarian_pred[0], hungarian_pred[1], match_dict)  # test hungarian method
                am_acc, am_num_tp, am_num_gt = calc_volume_accuracy(np.arange(cost_matrix.shape[0]), argmin_pred, match_dict)  # test argmin method

                num_gts_argmin += am_num_gt
                num_tps_argmin += am_num_tp
                num_gts_hungarian += hug_num_gt
                num_tps_hungarian += hug_num_tp
                if verbose:
                    print_log_message(f"Vol {batch_idx * dataloader.batch_size + i}: top-1 hungarian accuracy {hug_acc}, argmin accuracy {am_acc} | the num_gts is {am_num_gt}")

    hug_acc_all, hug_gt_mean = num_tps_hungarian / num_gts_hungarian, num_gts_hungarian / (len(dataloader.dataset) - 1)
    am_acc_all, am_gt_mean = num_tps_argmin / num_gts_argmin, num_gts_argmin / (len(dataloader.dataset) - 1)
    return hug_acc_all, am_acc_all, hug_gt_mean, am_gt_mean


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--name-reg', type = str, default = r"[iI]ma?ge?_?[sS]t(?:ac)?k_?\d+_dk?\d+.*[wW]\d+_?Dt\d{6}")
    parser.add_argument('--random-seed', type = int, default = 1024)
    parser.add_argument('--data-root', type = str, default = "/home/cbmi/wyx/CenDer_PLOS_CompBio/")
    # neuron recognition (train)
    parser.add_argument('--rec-fp16', action = "store_true")
    parser.add_argument('--rec-epoch', default = 400, type = int)
    parser.add_argument('--rec-num-workers', default = 1, type = int)
    parser.add_argument('--rec-batch-size', default = 32, type = int)
    parser.add_argument('--rec-reference-id', default = 16, type = int)
    parser.add_argument('--rec-model-load-path', type = str, default = "")
    # embedding method
    parser.add_argument('--rec-channel-base', type = int, default = 32)
    parser.add_argument('--rec-group-base', type = int, default = 4)
    parser.add_argument('--rec-len-embedding', type = int, default = 56)
    parser.add_argument('--rec-hypersphere-radius', type = int, default = 16)

    args = parser.parse_args("")
    # print_info_message(args)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    path = "/home/cbmi/wyx/CenDer_PLOS_CompBio/models/2022_05_24_21_59_52_Oreo_nh128_nl6_ft0_nm0_dataall_elam_0.1_rt0_pt0.bin"
    model = RecFormer_Test.load(path).to(torch.device('cuda:0'))
    model.eval()

    # benchmark leifer 2017
    dataset = Dataset_fDNC(glob(os.path.join(args.data_root, config_fdnc.benchmark_NeRVE['animals']['fea_vecs']['4fDNC'], "*.npy")), is_fp16 = args.rec_fp16, num_pool = args.rec_num_workers)
    dataloader = DataLoader(dataset, args.rec_batch_size, collate_fn = collate_fn_volumes, shuffle = False, drop_last = False, pin_memory = False, num_workers = args.rec_num_workers)
    accs = evaluate_benchmark(dataloader, model, refer_idx = args.rec_reference_id, verbose = False)
    print_info_message(f"NeRVE: hungarian method {accs[0] * 100:.2f} \t num_gts: {accs[2]:.2f} \t | \t argmin method {accs[1] * 100:.2f} \t num_gts: {accs[3]:.2f}")

    # benchmark NeuroPAL
    dataset = Dataset_fDNC(glob(os.path.join(args.data_root, config_fdnc.benchmark_NeuroPAL['animals']['fea_vecs']['4fDNC'], "*.npy")), is_fp16 = args.rec_fp16, num_pool = args.rec_num_workers)
    dataloader = DataLoader(dataset, args.rec_batch_size, collate_fn = collate_fn_volumes, shuffle = False, drop_last = False, pin_memory = False, num_workers = args.rec_num_workers)
    accs = np.array([evaluate_benchmark(dataloader, model, refer_idx = ref_idx) for ref_idx in range(len(dataset))])
    print_info_message(f"NeuroPAL: hungarian method {np.mean(accs[:, 0] * 100):.2f} ± {np.std(accs[:, 0] * 100):.2f} \t num_gts: {np.mean(accs[:, 2]):.2f} ± {np.std(accs[:, 2]):.2f} \t | \t"
                       f"argmin method {np.mean(accs[:, 1] * 100):.2f} ± {np.std(accs[:, 1] * 100):.2f} \t num_gts: {np.mean(accs[:, 3]):.2f} ± {np.std(accs[:, 3]):.2f}")

    # benchmark NeuroPAL Chaudhary
    dataset = Dataset_fDNC(glob(os.path.join(args.data_root, config_fdnc.benchmark_NeuroPAL_Chaudhary['animals']['fea_vecs']['4fDNC'], "*.npy")), is_fp16 = args.rec_fp16, num_pool = args.rec_num_workers)
    dataloader = DataLoader(dataset, args.rec_batch_size, collate_fn = collate_fn_volumes, shuffle = False, drop_last = False, pin_memory = False, num_workers = args.rec_num_workers)
    accs = np.array([evaluate_benchmark(dataloader, model, refer_idx = ref_idx) for ref_idx in range(len(dataset))])
    print_info_message(f"NeuroPAL Chaudhary: hungarian method {np.mean(accs[:, 0] * 100):.2f} ± {np.std(accs[:, 0] * 100):.2f} \t num_gts: {np.mean(accs[:, 2]):.2f} ± {np.std(accs[:, 2]):.2f} \t | \t"
                       f"argmin method {np.mean(accs[:, 1] * 100):.2f} ± {np.std(accs[:, 1] * 100):.2f} \t num_gts: {np.mean(accs[:, 3]):.2f} ± {np.std(accs[:, 3]):.2f}")

    # benchmark CeNDeR
    # dataset = Dataset_fDNC(glob(os.path.join(args.data_root, config_cender.dataset['animals']['fea_vecs']['test_root'], "*.npy")), is_fp16 = args.rec_fp16, num_pool = args.rec_num_workers)
    # dataloader = DataLoader(dataset, args.rec_batch_size, collate_fn = collate_fn_volumes, shuffle = False, drop_last = False, pin_memory = False, num_workers = args.rec_num_workers)
    # accs = evaluate_benchmark(dataloader, model, refer_idx = 9, verbose = False)
    # print_info_message(f"CeNDeR: hungarian method {accs[0] * 100:.2f} \t num_gts: {accs[2]:.2f} \t | \t argmin method {accs[1] * 100:.2f} \t num_gts: {accs[3]:.2f}")
