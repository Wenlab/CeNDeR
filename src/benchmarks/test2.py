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
from tqdm import tqdm
from collections import OrderedDict
from torch.nn import functional as F
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment

sys.path.append("/home/cbmi/wyx/CenDer_PLOS_CompBio")
from src.common_utils.prints import print_info_message, print_log_message
from src.benchmarks.datasets.CeNDeR import Dataset_CeNDeR
from src.recognition.inference.network import RecFuseNetworkLinear


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


def evaluate_benchmark_base(dataloader, model, refer_idx: int = 0, verbose: bool = False, test_batch_size: int = 32):
    model.eval()
    results = OrderedDict()
    results.update({name: [[], []] for name in dataloader.dataset.names})  # embedding, id
    with torch.inference_mode():
        for idx, (feas, ids, names, lens) in enumerate(dataloader):
            embeds, _ = model(feas.cuda(), mode = 1)
            for embed, name, _id, in zip(embeds.detach().cpu(), names, ids):
                results[name][0].append(embed)
                results[name][1].append(_id)
    keys = list(results.keys())
    batches = [keys[i: i + test_batch_size] for i in range(0, dataloader.dataset.num_vols, test_batch_size)]

    num_gts_argmin, num_tps_argmin = 0, 0
    num_gts_hungarian, num_tps_hungarian = 0, 0
    vol_result = dict()
    for batch_idx, batch_name in enumerate(batches):
        ref_name = batch_name[min(refer_idx, len(batches))]
        ref_embed = torch.vstack(results[ref_name][0])
        ref_id = torch.Tensor(results[ref_name][1]).numpy()
        for i, name in enumerate(batch_name):
            if name != ref_name:
                embed = torch.vstack(results[name][0])
                _id = torch.Tensor(results[name][1]).numpy()
                match_dict = {gt_m[0]: gt_m[1] for gt_m in find_match(_id, ref_id)[0]}
                cost_matrix = (1 - torch.mm(embed, ref_embed.transpose(1, 0))).numpy()

                argmin_pred = np.argmin(cost_matrix, axis = 1)  # 1-dim np.ndarray
                hungarian_pred = linear_sum_assignment(cost_matrix)  # 2-dim np.ndarray

                hug_acc, hug_num_tp, hug_num_gt = calc_volume_accuracy(hungarian_pred[0], hungarian_pred[1], match_dict)  # test by hungarian method
                am_acc, am_num_tp, am_num_gt = calc_volume_accuracy(np.arange(cost_matrix.shape[0]), argmin_pred, match_dict)  # test by argmin method

                num_gts_argmin += am_num_gt
                num_tps_argmin += am_num_tp
                num_gts_hungarian += hug_num_gt
                num_tps_hungarian += hug_num_tp
                vol_result[name] = hug_num_tp / hug_num_gt
                if verbose:
                    print_log_message(f"Vol {batch_idx * dataloader.batch_size + i}: top-1 hungarian accuracy {hug_acc}, argmin accuracy {am_acc} | the num_gts is {am_num_gt}")

    hug_acc_all, hug_gt_mean = num_tps_hungarian / num_gts_hungarian, num_gts_hungarian / (dataloader.dataset.num_vols - 1)
    am_acc_all, am_gt_mean = num_tps_argmin / num_gts_argmin, num_gts_argmin / (dataloader.dataset.num_vols - 1)
    return hug_acc_all, hug_gt_mean, num_tps_hungarian, num_gts_hungarian, am_acc_all, am_gt_mean, num_tps_argmin, num_gts_argmin, (dataloader.dataset.num_vols - 1)


def evaluate_benchmark(dataloader, model, refer_idx: int = 0, verbose: bool = False, test_batch_size: int = 32):
    hug_acc_all, hug_gt_mean, num_tps_hungarian, num_gts_hungarian, am_acc_all, am_gt_mean, num_tps_argmin, num_gts_argmin, num_vols = evaluate_benchmark_base(dataloader, model, refer_idx, verbose, test_batch_size)
    return hug_acc_all, am_acc_all, hug_gt_mean, am_gt_mean, num_tps_hungarian / num_vols


def evaluate_multi_worms_tracking(dataloaders, model, name_animals = None, refer_idx: int = 0, verbose: bool = False, test_batch_size: int = 32):
    res = np.array([evaluate_benchmark_base(dataloader, model, refer_idx, verbose, test_batch_size) for dataloader in dataloaders])
    if verbose:
        res_str = "".join([f"animal {name} hungarian: acc {(r[0] * 100.0):.2f} num_gts {r[1]:.2f}, \t argmin: acc {(r[4] * 100.0):.2f} num_gts {r[5]:.2f}" for name, r in zip(name_animals, res)])
        print_info_message(res_str)
    hug_acc, am_acc = res[:, 2].sum() / res[:, 3].sum(), res[:, 6].sum() / res[:, 7].sum()
    hug_mean_gts, am_mean_gts = res[:, 3].sum() / res[:, 8].sum(), res[:, 7].sum() / res[:, 8].sum()
    return hug_acc, hug_mean_gts, am_acc, am_mean_gts, res[:, 2].sum() / res[:, 8].sum()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--name-reg', type = str, default = r"[iI]ma?ge?_?[sS]t(?:ac)?k_?\d+_dk?\d+.*[wW]\d+_?Dt\d{6}")
    parser.add_argument('--random-seed', type = int, default = 1024)
    parser.add_argument('--data-root', type = str, default = "/home/cbmi/wyx/CenDer_PLOS_CompBio/")
    # neuron recognition (train)
    parser.add_argument('--rec-fp16', action = "store_true")
    parser.add_argument('--rec-num-workers', default = 1, type = int)
    parser.add_argument('--rec-batch-size', default = 256, type = int)
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

    # path = "/home/cbmi/wyx/CenDer_PLOS_CompBio/models/rec_model.ckpt"
    path = "/home/cbmi/wyx/CenDer_PLOS_CompBio/models/released/cender_rec_model.ckpt"
    model = RecFuseNetworkLinear(input_dim = (75, 80), output_dim = 56, num_ids = 164, channel_base = 32, group_base = 4, dropout_ratio = 0.2, activation_method = "celu").cuda()
    model.load_state_dict(torch.load(path, map_location = 'cuda:0')['network'])
    model = model.half() if args.rec_fp16 else model
    model.eval()

    # benchmark leifer 2017
    dataset = Dataset_CeNDeR(glob(os.path.join(args.data_root, "data/benchmarks/supp/e1/test_tracking", "*.npy")), is_fp16 = args.rec_fp16, num_pool = args.rec_num_workers)
    dataloader = DataLoader(dataset, args.rec_batch_size, shuffle = False, drop_last = False, pin_memory = False, num_workers = args.rec_num_workers)
    accs = evaluate_benchmark(dataloader, model, refer_idx = args.rec_reference_id, verbose = False)
    # print_info_message(f"NeRVE: hungarian method {accs[0] * 100:.2f} \t num_gts: {accs[2]:.2f} \t | \t argmin method {accs[1] * 100:.2f} \t num_gts: {accs[3]:.2f}")
    print_info_message(f"NeRVE: accuracy:  {accs[0] * 100:.2f} \t num_gts: {accs[2]:.2f} \t num_hits: {(accs[4]):.2f}")

    # benchmark NeuroPAL
    dataset = Dataset_CeNDeR(glob(os.path.join(args.data_root, "data/benchmarks/supp/e1/test_neuropal_our", "*.npy")), is_fp16 = args.rec_fp16, num_pool = args.rec_num_workers)
    dataloader = DataLoader(dataset, args.rec_batch_size, shuffle = False, drop_last = False, pin_memory = False, num_workers = args.rec_num_workers)
    accs = np.array([evaluate_benchmark(dataloader, model, refer_idx = ref_idx) for ref_idx in range(dataset.num_vols)])
    # print_info_message(f"NeuroPAL: hungarian method {np.mean(accs[:, 0] * 100):.2f} ± {np.std(accs[:, 0] * 100):.2f} \t num_gts: {np.mean(accs[:, 2]):.2f} ± {np.std(accs[:, 2]):.2f} \t | \t"
    #                    f"argmin method {np.mean(accs[:, 1] * 100):.2f} ± {np.std(accs[:, 1] * 100):.2f} \t num_gts: {np.mean(accs[:, 3]):.2f} ± {np.std(accs[:, 3]):.2f}")
    print_info_message(f"NeuroPAL Yu: accuracy: {np.mean(accs[:, 0] * 100):.2f} ± {np.std(accs[:, 0] * 100):.2f} \t num_gts: {np.mean(accs[:, 2]):.2f} ± {np.std(accs[:, 2]):.2f} \t "
                       f"num_hits: {np.mean(accs[:, 4]):.2f} ± {np.std(accs[:, 4]):.2f}")

    # benchmark NeuroPAL Chaudhary
    dataset = Dataset_CeNDeR(glob(os.path.join(args.data_root, "data/benchmarks/supp/e1/test_neuropal_Chaudhary", "*.npy")), is_fp16 = args.rec_fp16, num_pool = args.rec_num_workers)
    dataloader = DataLoader(dataset, args.rec_batch_size, shuffle = False, drop_last = False, pin_memory = False, num_workers = args.rec_num_workers)
    accs = np.array([evaluate_benchmark(dataloader, model, refer_idx = ref_idx) for ref_idx in range(dataset.num_vols)])
    print_info_message(f"NeuroPAL Chaudhary: accuracy: {np.mean(accs[:, 0] * 100):.2f} ± {np.std(accs[:, 0] * 100):.2f} \t num_gts: {np.mean(accs[:, 2]):.2f} ± {np.std(accs[:, 2]):.2f} \t "
                       f"num_hits: {np.mean(accs[:, 4]):.2f} ± {np.std(accs[:, 4]):.2f}")

    # benchmark CeNDeR within
    dataset = Dataset_CeNDeR(glob(os.path.join(args.data_root, "data/benchmarks/CeNDeR/base/C1", "*.npy")), is_fp16 = args.rec_fp16, num_pool = args.rec_num_workers)
    dataloader = DataLoader(dataset, args.rec_batch_size)
    accs = evaluate_benchmark(dataloader, model, refer_idx = 0, verbose = False, )
    # print_info_message(f"CeNDeR within: hungarian method {accs[0] * 100:.2f} \t num_gts: {accs[2]:.2f} \t | \t argmin method {accs[1] * 100:.2f} \t num_gts: {accs[3]:.2f}")
    print_info_message(f"CeNDeR within: accuracy:  {accs[0] * 100:.2f} \t num_gts: {accs[2]:.2f} \t num_hits: {(accs[4]):.2f}")

    # benchmark CeNDeR across
    dataloaders = [DataLoader(Dataset_CeNDeR(glob(os.path.join(args.data_root, "data/benchmarks/CeNDeR/base", n, "*.npy")), args.rec_fp16, args.rec_num_workers), args.rec_batch_size) for n in ["C2", "C3"]]
    accs = evaluate_multi_worms_tracking(dataloaders, model, refer_idx = 0, verbose = False)
    print_info_message(f"CeNDeR across: hungarian method {accs[0] * 100:.2f} \t num_gts: {accs[1]:.2f} \t num_hits: {(accs[4]):.2f}")
