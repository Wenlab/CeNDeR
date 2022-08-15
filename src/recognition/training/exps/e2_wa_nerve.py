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
import torch.nn as nn
from glob import glob
from typing import List
from torch.optim import optimizer
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))))
from src.common_utils.prints import print_info_message, get_checkpoint_timestamp
from src.common_utils.metric.rec import top_1_accuracy_score, top_k_accuracy, top_1_accuracy_score_torch
from src.benchmarks.test2 import evaluate_benchmark, evaluate_multi_worms_tracking
from src.benchmarks.datasets.CeNDeR import Dataset_CeNDeR
from src.recognition.training.configs import e2_wa_nerve as cfg
from src.recognition.inference.dataset import RecFeatureDataset
from src.recognition.inference.network import RecFuseNetworkLinear, RecMarginalCosLossNetwork


def test_one_epoch(dataloader, test_names, num_ids, model, batch_size, is_fp16, is_h = False):
    """Return every volume result and all"""
    names, labels, preds = list(), list(), torch.zeros((len(dataloader.dataset), num_ids), dtype = torch.float16 if is_fp16 else torch.float32).cuda()
    model.eval()
    with torch.no_grad():
        for idx, (name, feas, ids) in enumerate(dataloader):
            preds[idx * batch_size: idx * batch_size + len(ids)] = model(feas.cuda(), mode = 2)
            labels.extend(ids.tolist())
            names.extend([vol_name for vol_name in name])
    preds, labels = np.array(preds.cpu()), np.array(labels)
    sum_result, top3sum_result = top_1_accuracy_score(labels, preds), top_k_accuracy(labels, preds, k = 3)
    vol_info = {vol_name: [[], []] for vol_name in test_names}

    for name, pred, label in zip(names, preds, labels):
        vol_info[name][0].append(pred)
        vol_info[name][1].extend(label)

    if is_h:
        vol_results = dict()
        num_gts_hungarian, num_tps_hungarian = 0, 0
        for vol_name, (vol_pred, vol_label) in vol_info.items():
            nd_pred, nd_label = np.array(vol_pred), np.array(vol_label)
            idx = np.argwhere(nd_label != cfg.id_map[-1])

            if nd_pred.shape[0] - nd_pred.shape[1] >= 0:
                cost_matrix = np.pad(1 - nd_pred, ((0, 0), (0, nd_pred.shape[0] - nd_pred.shape[1])), mode = 'edge')
            else:
                cost_matrix = 1 - nd_pred
            hungarian_pred = linear_sum_assignment(cost_matrix)  # 2-dim np.ndarray
            np.clip(hungarian_pred[1], a_min = 0, a_max = nd_pred.shape[1] - 1, out = hungarian_pred[1])

            tp = np.sum(hungarian_pred[1][idx] == nd_label[idx])
            gt = len(idx)
            num_tps_hungarian += tp
            num_gts_hungarian += gt
            vol_results[vol_name] = tp / gt
        sum_result_hungarian = num_tps_hungarian / num_gts_hungarian
    else:
        sum_result_hungarian = 0.0
        num_tps_hungarian = 0
        vol_results = {vol_name: top_1_accuracy_score(np.array(vol_label), np.array(vol_pred)) for vol_name, (vol_pred, vol_label) in vol_info.items()}

    return vol_results, sum_result, top3sum_result, sum_result_hungarian, num_tps_hungarian / len(test_names)


def train_val_procedure(model: nn.Module, criterion: nn.Module,
                        train_dataloader: DataLoader, val_dataloader: DataLoader,
                        test_dataloader: DataLoader, test_names: List,
                        optimizer: optimizer.Optimizer,
                        model_save_path: str,
                        tb_writer: SummaryWriter,
                        num_ids: int,
                        num_epochs: int = 200,
                        **kwargs):
    print_info_message(f"------------------------------------------- >>>>>>>>>>>>>>>>>>>>>>>>>>>> Training procedure starts! \n")
    results = [dict(), dict()]
    best_epoch = 0
    best_accuracy = [0.0, 0.0]
    train_accs = torch.zeros((len(train_dataloader)), dtype = torch.float16 if kwargs['is_fp16'] else torch.float32).cuda()
    val_accs = torch.zeros((len(val_dataloader)), dtype = torch.float16 if kwargs['is_fp16'] else torch.float32).cuda()

    for epoch in range(num_epochs):
        # =============== train ===============
        model.train()
        losses = 0.0
        for iter_idx, rs in enumerate(train_dataloader):
            ns, feas, ids = rs[0], rs[1].cuda(), rs[2].squeeze().cuda()
            # calculating
            optimizer.zero_grad()
            embedding, cos_dis = model(feas, mode = 1)
            loss = criterion(cos_dis, ids)
            loss.backward()
            optimizer.step()
            # recording
            losses += loss.item()
            train_accs[iter_idx] = top_1_accuracy_score_torch(ids, cos_dis)

        results[0][epoch] = [torch.mean(train_accs)]
        # =============== verification ===============
        model.eval()
        with torch.no_grad():
            rec_preds, rec_labels = list(), list()
            for iter_idx, rs in enumerate(val_dataloader):
                # calculating
                ns, feas, ids = rs[0], rs[1].cuda(), rs[2].squeeze().cuda()
                embedding, cos_dis = model(feas, mode = 1)
                # recording
                val_accs[iter_idx] = top_1_accuracy_score_torch(ids, cos_dis)
                # rec_preds.extend(cos_dis.cpu().tolist())
                rec_labels.extend(ids.tolist())

        results[1][epoch] = [torch.mean(val_accs)]
        tb_writer.add_scalar('train_loss', losses / len(train_dataloader), global_step = epoch)
        tb_writer.add_scalars('top1_accuracy', {"train": results[0][epoch][0] * 100.0, "val": results[1][epoch][0] * 100.0}, global_step = epoch)

        if best_accuracy[0] <= (results[1][epoch][0]):
            best_epoch, best_accuracy = epoch, results[1][epoch]
            torch.save({'network': model.state_dict()}, model_save_path)
            # test set
            # vol_results, sum_result, top3_result, _ = test_one_epoch(test_dataloader, test_names, num_ids, model, kwargs['batch_size'], kwargs['is_fp16'], is_h = False)
            # tb_writer.add_scalars('test_accuracy', {"argmin-top1": sum_result * 100.0}, global_step = epoch)
            vol_results, sum_result, top3_result, sum_result_hungarian, _ = test_one_epoch(test_dataloader, test_names, num_ids, model, kwargs['batch_size'], kwargs['is_fp16'], is_h = True)
            tb_writer.add_scalars('test_accuracy', {"hung": sum_result_hungarian * 100.0}, global_step = epoch)

    tb_writer.close()
    print_info_message(f"------------------------------------------- >>>>>>>>>>>>>>>>>>>>>>>>>>>>"
                       f"The best val epoch of {kwargs['checkpoint_timestamp']} model is {best_epoch} \t  Top-1 accuracy: {(best_accuracy[0] * 100.0):.2f} % \t ")
    # test
    model.load_state_dict(torch.load(model_save_path, map_location = 'cuda:0')['network'])
    model.eval()
    vol_results, sum_result, top3_result, sum_result_hungarian, num_hits = test_one_epoch(test_dataloader, test_names, num_ids, model, kwargs['batch_size'], kwargs['is_fp16'], is_h = True)
    print_info_message(f"NeRVE Testset accuracy: {(sum_result_hungarian * 100.0):.2f} \t num_hits: {num_hits:.2f}")
    # res = {name.split("/")[-1][:-4]: -100 for name in sorted(all_paths)}
    # for n, v in vol_results.items():
    #     res[n] = v
    # print(list(res.values()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--name-reg', type = str, default = r"[iI]ma?ge?_?[sS]t(?:ac)?k_?\d+_dk?\d+.*[wW]\d+_?Dt\d{6}")
    parser.add_argument('--data-root', type = str, default = "")
    parser.add_argument('--random-seed', type = int, default = 999)
    parser.add_argument('--checkpoint-timestamp', type = str, default = f"{get_checkpoint_timestamp()}_{np.random.randint(100000)}")
    # neuron recognition (train)
    parser.add_argument('--rec-tflag', default = 130, type = int)
    parser.add_argument('--rec-shuffle', default = 1, type = int)
    parser.add_argument('--rec-fp16', action = "store_true")
    parser.add_argument('--rec-epoch', default = 150, type = int)
    parser.add_argument('--rec-num-workers', default = 8, type = int)
    parser.add_argument('--rec-batch-size', default = 256, type = int)
    parser.add_argument('--rec-model-load-path', type = str, default = "")
    parser.add_argument('--rec-model-save-path', type = str, default = 'models/supp/e2_wa_nerve')
    # embedding method
    parser.add_argument('--rec-channel-base', type = int, default = 32)
    parser.add_argument('--rec-group-base', type = int, default = 4)
    parser.add_argument('--rec-len-embedding', type = int, default = 56)
    parser.add_argument('--rec-hypersphere-radius', type = int, default = 32)
    parser.add_argument('--rec-loss-coefficients', type = float, nargs = "+", default = [1.05, 0, 0])
    # tensorboard
    parser.add_argument('--rec-tensorboard-root', type = str, default = "tb_log/supp/e2_wa_nerve")

    args = parser.parse_args()
    print_info_message(args)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    # ----- data preparation --------------------
    all_paths = sorted(list(glob(os.path.join(args.data_root, cfg.dataset['animals']['root']['fea_vecs'], "*.npy"))))
    training_flag = cfg.dataset['animals']['set']['train'][1] if args.rec_tflag == 0 else args.rec_tflag
    val_flag = training_flag + 20
    if args.rec_shuffle:
        np.random.shuffle(all_paths)

    ds = Dataset_CeNDeR(all_paths[:training_flag], is_fp16 = args.rec_fp16, num_pool = args.rec_num_workers)
    train_dataset = RecFeatureDataset(Xs = ds.Xs[:, 3:], ys = [cfg.id_map[i] if i in cfg.pi else len(cfg.pi) for i in ds.ys], names = ds.pad_name, is_train = True, is_fp16 = args.rec_fp16)
    train_dataloader = DataLoader(train_dataset, batch_size = args.rec_batch_size, drop_last = True, shuffle = True, num_workers = args.rec_num_workers)

    ds = Dataset_CeNDeR(all_paths[training_flag:val_flag], is_fp16 = args.rec_fp16, num_pool = args.rec_num_workers)
    val_dataset = RecFeatureDataset(Xs = ds.Xs[:, 3:], ys = [cfg.id_map[i] if i in cfg.pi else len(cfg.pi) for i in ds.ys], names = ds.pad_name, is_train = True, is_fp16 = args.rec_fp16)
    val_dataloader = DataLoader(val_dataset, batch_size = args.rec_batch_size, drop_last = True, shuffle = True, num_workers = args.rec_num_workers)

    tds = Dataset_CeNDeR(all_paths[val_flag:], is_fp16 = args.rec_fp16, num_pool = args.rec_num_workers)
    test_dataset = RecFeatureDataset(Xs = tds.Xs[:, 3:], ys = [cfg.id_map[i] if i in cfg.pi else len(cfg.pi) for i in tds.ys], names = tds.pad_name, is_train = False, is_fp16 = args.rec_fp16)
    test_dataloader = DataLoader(test_dataset, batch_size = args.rec_batch_size, drop_last = False, shuffle = False, num_workers = args.rec_num_workers)

    # ----- network --------------------
    model = RecFuseNetworkLinear(input_dim = (cfg.method['fea_vecs_setup']['rec_knn_k'] * 3, cfg.method['fea_vecs_setup']['rec_des_len'] * 4),
                                 output_dim = args.rec_len_embedding, num_ids = len(cfg.id_map), channel_base = args.rec_channel_base,
                                 group_base = args.rec_group_base, dropout_ratio = 0.2, activation_method = "celu").cuda()
    if os.path.isfile(args.rec_model_load_path):
        model.load_state_dict(torch.load(args.rec_model_load_path, map_location = 'cuda:0')['network'])
    model = model.half() if args.rec_fp16 else model

    criterion = RecMarginalCosLossNetwork(len_embedding = args.rec_len_embedding, coefficients = args.rec_loss_coefficients, hypersphere_radius = args.rec_hypersphere_radius).cuda()
    criterion = criterion.half() if args.rec_fp16 else criterion
    os.makedirs(os.path.join(args.rec_model_save_path), exist_ok = True)

    train_val_procedure(model = model,
                        criterion = criterion,
                        train_dataloader = train_dataloader,
                        val_dataloader = val_dataloader,
                        test_dataloader = test_dataloader,
                        test_names = tds.names,
                        batch_size = args.rec_batch_size,
                        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3),
                        is_fp16 = args.rec_fp16,
                        num_ids = len(cfg.id_map),
                        hr = args.rec_hypersphere_radius,
                        num_epochs = args.rec_epoch,
                        checkpoint_timestamp = args.checkpoint_timestamp,
                        tb_writer = SummaryWriter(os.path.join(args.rec_tensorboard_root, args.checkpoint_timestamp)),
                        model_save_path = os.path.join(args.rec_model_save_path, f"{args.checkpoint_timestamp}.ckpt"),
                        )

    # ==========  evaluate benchmarks  ====================
    batch_size = 32
    model.eval()

    # benchmark NeuroPAL
    dataset = Dataset_CeNDeR(glob(os.path.join(args.data_root, cfg.benchmark_NeuroPAL['animals']['fea_vecs'], "*.npy")), is_fp16 = args.rec_fp16, num_pool = args.rec_num_workers)
    dataloader = DataLoader(dataset, args.rec_batch_size)
    accs = np.array([evaluate_benchmark(dataloader, model, refer_idx = ref_idx) for ref_idx in range(dataset.num_vols)])
    print_info_message(f"NeuroPAL Yu: accuracy: {np.mean(accs[:, 0] * 100):.2f} ± {np.std(accs[:, 0] * 100):.2f} \t num_gts: {np.mean(accs[:, 2]):.2f} ± {np.std(accs[:, 2]):.2f} \t "
                       f"num_hits: {np.mean(accs[:, 4]):.2f} ± {np.std(accs[:, 4]):.2f}")

    # benchmark NeuroPAL Chaudhary
    dataset = Dataset_CeNDeR(glob(os.path.join(args.data_root, cfg.benchmark_NeuroPAL_Chaudhary['animals']['fea_vecs'], "*.npy")), is_fp16 = args.rec_fp16, num_pool = args.rec_num_workers)
    dataloader = DataLoader(dataset, args.rec_batch_size)
    accs = np.array([evaluate_benchmark(dataloader, model, refer_idx = ref_idx) for ref_idx in range(dataset.num_vols)])
    print_info_message(f"NeuroPAL Chaudhary: accuracy: {np.mean(accs[:, 0] * 100):.2f} ± {np.std(accs[:, 0] * 100):.2f} \t num_gts: {np.mean(accs[:, 2]):.2f} ± {np.std(accs[:, 2]):.2f} \t "
                       f"num_hits: {np.mean(accs[:, 4]):.2f} ± {np.std(accs[:, 4]):.2f}")

    # benchmark CeNDeR
    dataloaders = [DataLoader(Dataset_CeNDeR(glob(os.path.join(args.data_root, "data/benchmarks/CeNDeR/feavec4region", n, "*.npy")), args.rec_fp16, args.rec_num_workers), batch_size) for n in ["C1", "C2", "C3"]]
    accs = evaluate_multi_worms_tracking(dataloaders, model, refer_idx = 0, verbose = False)
    print_info_message(f"CeNDeR: hungarian method {accs[0] * 100:.2f} \t num_gts: {accs[1]:.2f} \t num_hits: {(accs[4]):.2f}")
