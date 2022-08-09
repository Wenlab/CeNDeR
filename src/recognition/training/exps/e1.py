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
from multiprocessing import Pool
from torch.optim import optimizer
from collections import OrderedDict
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))))
from src.common_utils.dataset_building import split_multi_individuals_datasets
from src.common_utils.prints import get_checkpoint_timestamp, print_log_message
from src.preprocessing.training.utils import extract_annos
from src.recognition.training.configs import e1 as cfg
from src.recognition.training.extract_training_data import extract_preprocessing_json, select_ids
from src.recognition.inference.feature_maker.load_features import make_one_volume_neuronal_features
from src.common_utils.prints import print_info_message
from src.common_utils.metric.rec import top_1_accuracy_score, top_k_accuracy, top_1_accuracy_score_torch
from src.recognition.training.extract_training_data import neurons2data
from src.recognition.inference.dataset import RecFeatureDataset
from src.recognition.inference.network import RecFuseNetworkLinear, RecMarginalCosLossNetwork


# ==========  within across  ====================
def make_across_testset(labels):
    Xs, ys, names, name_list = list(), list(), list(), list()
    for name, vol_label in labels.items():
        name_list.append(name)
        for neuron_id, neuron_X in vol_label.items():
            Xs.append(neuron_X)
            ys.append(neuron_id)
            names.append(name)
    return Xs, ys, names, name_list


def calc_volume_accuracy(ref, vol, match_dict: dict):
    n_gt, n_tp = 0, 0
    for r_idx, r in enumerate(ref):
        if r in match_dict:
            n_gt += 1
            if match_dict[r] == vol[r_idx]:
                n_tp += 1
    acc = n_tp / n_gt
    return acc, n_tp, n_gt


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


def test_across4one_animal(model, dataloader, vol_names, refer_idx: int = 0, verbose: bool = False, test_batch_size: int = 32):
    model.eval()
    results = OrderedDict()
    results.update({name: [[], []] for name in vol_names})  # embedding, id
    with torch.inference_mode():
        for idx, (names, feas, ids) in enumerate(dataloader):
            embeds, _ = model(feas.cuda(), mode = 1)
            for embed, name, _id, in zip(embeds.detach().cpu(), names, ids):
                results[name][0].append(embed)
                results[name][1].append(_id)
    keys = list(results.keys())
    batches = [keys[i: i + test_batch_size] for i in range(0, len(vol_names), test_batch_size)]

    num_gts_argmin, num_tps_argmin = 0, 0
    num_gts_hungarian, num_tps_hungarian = 0, 0
    vol_result = {}
    for batch_idx, batch_name in enumerate(batches):
        ref_name = batch_name[refer_idx]
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
                    print_log_message(f"Vol {batch_idx * dataloader.batch_size + i}: top-1 hungarian accuracy {(hug_acc * 100.0):.2f}, the num_gts is {am_num_gt}")
    num_cal_vols = len(vol_names) - 1
    hug_acc_all, hug_gt_mean = num_tps_hungarian / num_gts_hungarian, num_gts_hungarian / num_cal_vols
    am_acc_all, am_gt_mean = num_tps_argmin / num_gts_argmin, num_gts_argmin / num_cal_vols
    return hug_acc_all, hug_gt_mean, num_tps_hungarian, num_gts_hungarian, am_acc_all, am_gt_mean, num_tps_argmin, num_gts_argmin, num_cal_vols


def test_across4multi_animals(model, dataloaders, name_animals, vol_names_list, refer_idx: int = 0, verbose: bool = False, test_batch_size: int = 32):
    res = np.array([test_across4one_animal(model, dataloader, vol_names, refer_idx, verbose, test_batch_size) for name, vol_names, dataloader in zip(name_animals, vol_names_list, dataloaders)])
    if verbose:
        res_str = "".join([f"animal {name} hungarian: acc {(r[0] * 100.0):.2f} num_gts {r[1]:.2f}, \t argmin: acc {(r[4] * 100.0):.2f} num_gts {r[5]:.2f}" for name, r in zip(name_animals, res)])
        print_info_message(res_str)
    hug_acc, am_acc = res[:, 2].sum() / res[:, 3].sum(), res[:, 6].sum() / res[:, 7].sum()
    hug_mean_gts, am_mean_gts = res[:, 3].sum() / res[:, 8].sum(), res[:, 7].sum() / res[:, 8].sum()
    return hug_acc, hug_mean_gts, am_acc, am_mean_gts, res[:, 2].sum() / res[:, 8].sum()


# ==========  within test  ====================
def test_within(dataloader, test_names, num_ids, model, batch_size, is_fp16, processing_ids, is_h = False):
    """Return every volume result and all"""
    names, labels, preds = list(), list(), torch.zeros((len(dataloader.dataset), num_ids), dtype = torch.float16 if is_fp16 else torch.float32).cuda()
    model.eval()
    with torch.no_grad():
        for idx, (name, feas, ids) in enumerate(dataloader):
            preds[idx * batch_size: idx * batch_size + len(ids)] = model(feas.cuda(), mode = 2)
            labels.extend(ids.tolist())
            names.extend([vol_name for vol_name in name[0]])
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
            idx = np.argwhere(nd_label != processing_ids[-1])

            if nd_pred.shape[0] - nd_pred.shape[1] >= 0:
                cost_matrix = np.pad(1.0 - nd_pred, ((0, 0), (0, nd_pred.shape[0] - nd_pred.shape[1])), mode = 'edge')
            else:
                cost_matrix = 1.0 - nd_pred
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

    return vol_results, sum_result, top3sum_result, sum_result_hungarian, num_tps_hungarian / len(vol_info)


# ==========  training  ====================
def make_fea_vecs(param):
    key, volume_3d_result, ccords, args, mode = param
    result = {i: fea for i, _, fea in zip(*make_one_volume_neuronal_features(key, volume_3d_result, ccords, mode = mode, args = args))}
    return key, result


def make_fea_multiprocess(labels, vols_ccords, args, mode: int = 0):
    params = [[key, labels[key], vols_ccords[key], args, mode] for key in labels.keys()]
    feas = {key: None for key in sorted(list(labels.keys()))}
    with Pool(min(16, len(params))) as p:
        for key, result in p.imap_unordered(make_fea_vecs, params):
            feas[key] = result
    return feas


def train_val_procedure(model: nn.Module, criterion: nn.Module,
                        train_dataloader: DataLoader, val_dataloader: DataLoader,
                        within_test_dataloader: DataLoader,
                        across_test_dataloaders,
                        across_test_vol_names,
                        cfg, test_vols,
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
            # embs = list()  # recording embedding into tensorboard
            rec_preds, rec_labels = list(), list()
            for iter_idx, rs in enumerate(val_dataloader):
                # calculating
                ns, feas, ids = rs[0], rs[1].cuda(), rs[2].squeeze().cuda()
                embedding, cos_dis = model(feas, mode = 1)
                # recording
                # embs.append(embedding)
                val_accs[iter_idx] = top_1_accuracy_score_torch(ids, cos_dis)
                rec_labels.extend(ids.tolist())

        results[1][epoch] = [torch.mean(val_accs)]
        tb_writer.add_scalar('train_loss', losses / len(train_dataloader), global_step = epoch)
        tb_writer.add_scalars('top1_accuracy', {"train": results[0][epoch][0] * 100.0, "val": results[1][epoch][0] * 100.0}, global_step = epoch)

        if best_accuracy[0] <= (results[1][epoch][0]):
            best_epoch, best_accuracy = epoch, results[1][epoch]
            torch.save({'network': model.state_dict()}, model_save_path)
            # print_info_message(f"Model has been saved in {model_save_path} !")
            # tb_writer.add_embedding(mat = F.normalize(torch.cat(embs, dim = 0), dim = 1) * kwargs['hr'], metadata = rec_labels, global_step = epoch, tag = "samples")
            # tb_writer.add_embedding(mat = F.normalize(model.norm_linear.weight, dim = 1) * kwargs['hr'], metadata = kwargs['processing_ids'], global_step = epoch, tag = "centers")
            # test set
            vol_results, sum_result, top3_result, _, _ = test_within(within_test_dataloader, test_vols, num_ids, model, kwargs['batch_size'], kwargs['is_fp16'], kwargs['processing_ids'], is_h = False)
            hug_acc, hug_mean_gts, am_acc, am_mean_gts, _ = test_across4multi_animals(model, across_test_dataloaders, cfg.dataset['animals']['names'][1:], across_test_vol_names)
            tb_writer.add_scalars('test_accuracy', {"within_argmin": sum_result * 100.0, "across_hung": hug_acc * 100.0}, global_step = epoch)

    print_info_message(f"------------------------------------------- >>>>>>>>>>>>>>>>>>>>>>>>>>>> Training procedure finished! \n"
                       f"The best val epoch of {kwargs['checkpoint_timestamp']} model is {best_epoch} \t "
                       f"Top-1 accuracy: {(best_accuracy[0] * 100.0):.2f} % \t ")
    tb_writer.close()

    # test
    model.load_state_dict(torch.load(model_save_path, map_location = 'cuda:0')['network'])
    model.eval()
    vol_results, sum_result, top3_result, sum_result_hungarian, num_hits = test_within(within_test_dataloader, test_vols, num_ids, model, kwargs['batch_size'], kwargs['is_fp16'], kwargs['processing_ids'], is_h = True)
    print_info_message(f"Within testset: accuracy: {(sum_result_hungarian * 100.0):.2f} \t num_hits: {num_hits:.2f}")
    # res = {name: -100 for name in within_labels.keys()}
    # for name, r in vol_results.items():
    #     res[name] = r
    # print(list(res.values()))
    hug_acc, hug_mean_gts, am_acc, am_mean_gts, num_hits = test_across4multi_animals(model, across_test_dataloaders, cfg.dataset['animals']['names'][1:], across_test_vol_names)
    print_info_message(f"Across testset: {len(cfg.dataset['animals']['names'][1:])} animals \t accuracy: {hug_acc * 100:.2f} \t num_gts: {hug_mean_gts:.2f} \t num_hits: {num_hits:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--name-reg', type = str, default = r"[iI]ma?ge?_?[sS]t(?:ac)?k_?\d+_dk?\d+.*[wW]\d+_?Dt\d{6}")
    parser.add_argument('--load-preprocess-result-root', type = str, default = "data/dataset/proofreading")
    parser.add_argument('--data-root', type = str, default = "")
    parser.add_argument('--random-seed', type = int, default = 520)
    parser.add_argument('--checkpoint-timestamp', type = str, default = f"{get_checkpoint_timestamp()}_{np.random.randint(100000)}")
    parser.add_argument('--label-root', type = str, default = "data/dataset/label")
    # neural points setting
    parser.add_argument('--rec-tflag', default = 130, type = int)
    parser.add_argument('--rec-fea-mode', default = 0, type = int)
    parser.add_argument('--rec-others-class', default = 1, type = int)
    parser.add_argument('--rec-xoy-unit', type = float, default = 0.3, help = "um/pixel")
    parser.add_argument('--rec-z-unit', type = float, default = 1.5, help = "um/pixel")
    parser.add_argument('--rec-worm-diagonal-line', type = float, default = 400.0)
    # knn feature
    parser.add_argument('--rec-knn-k', type = int, default = 25)
    # neural density feature
    parser.add_argument('--rec-des-len', type = int, default = 20)
    # neuron recognition (train)
    parser.add_argument('--rec-fp16', action = "store_true")
    parser.add_argument('--rec-epoch', default = 300, type = int)
    parser.add_argument('--rec-num-workers', default = 8, type = int)
    parser.add_argument('--rec-batch-size', default = 256, type = int)
    parser.add_argument('--rec-model-load-path', type = str, default = "")
    parser.add_argument('--rec-model-save-path', type = str, default = "models/supp/e1")
    parser.add_argument('--rec-shuffle', default = 1, type = int)

    # embedding method
    parser.add_argument('--rec-channel-base', type = int, default = 32)
    parser.add_argument('--rec-group-base', type = int, default = 4)
    parser.add_argument('--rec-len-embedding', type = int, default = 56)
    parser.add_argument('--rec-hypersphere-radius', type = int, default = 32)
    parser.add_argument('--rec-loss-coefficients', type = float, nargs = "+", default = [1.05, 0.0, 0.05])
    # tensorboard
    parser.add_argument('--rec-tensorboard-root', type = str, default = "tb_log/supp/e1")

    args = parser.parse_args()
    args.rec_z_scale = args.rec_z_unit / args.rec_xoy_unit
    print_info_message(args)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    # ----- data preparation --------------------
    # vols_ccords: {vol_name: [[xmin, ymin], mass_of_center, anterior_y, posterior_y, ventral_x, dorsal_x]}
    vols_xymin, vols_ccords = extract_preprocessing_json(os.path.join(args.load_preprocess_result_root, "*/*.json"))

    labels = [{k: v for file_name, idxes in idv_label for k, v in extract_annos(os.path.join(args.label_root, file_name), idxes, args.name_reg).items()}
              for idv_name, idv_label in cfg.dataset['animals']['label'].items()]
    labels = [{k: {i: [[p[1] - vols_xymin[k][0], p[2] - vols_xymin[k][1], p[3] - vols_xymin[k][0], p[4] - vols_xymin[k][1], p[0]] for p in pp] for i, pp in vol.items()}
               for k, vol in idv_labels.items()} for idv_labels in labels]

    # ==========  within dataset  ====================
    t_idx = args.rec_tflag
    val_idx = t_idx + 20
    processing_ids = select_ids(labels[0], 225)[0]
    within_labels, dataset_names = split_multi_individuals_datasets(labels = labels[:1], indexes = [[(0, t_idx), (t_idx, val_idx), (val_idx, len(labels[0]))]], shuffle_type = args.rec_shuffle)
    # ----- samples making --------------------
    within_engineering_feature = make_fea_multiprocess(within_labels, vols_ccords, args, mode = args.rec_fea_mode)
    trains, vals, tests, num_ids, id_map, processing_ids = neurons2data(within_engineering_feature.copy(), dataset_names = dataset_names,
                                                                        include_others_class = args.rec_others_class, given_ids = sorted(processing_ids), verbose = False)
    # ----- dataloader --------------------
    train_dataset = RecFeatureDataset(Xs = trains[0], ys = trains[1], names = trains[2], is_train = True, is_fp16 = args.rec_fp16)
    train_dataloader = DataLoader(train_dataset, batch_size = args.rec_batch_size, drop_last = True, shuffle = True, pin_memory = False, num_workers = 1)
    val_dataset = RecFeatureDataset(Xs = vals[0], ys = vals[1], names = vals[2], is_train = False, is_fp16 = args.rec_fp16)
    val_dataloader = DataLoader(val_dataset, batch_size = args.rec_batch_size, drop_last = False, shuffle = True, pin_memory = False, num_workers = 1)
    within_test_dataset = RecFeatureDataset(tests[0], tests[1], tests[2], is_train = False, is_fp16 = args.rec_fp16)
    within_test_dataloader = DataLoader(within_test_dataset, batch_size = args.rec_batch_size, drop_last = False, shuffle = True, pin_memory = False, num_workers = 1)

    # ==========  across dataset  ====================
    across_test_infos = [make_across_testset(make_fea_multiprocess(idv_label, vols_ccords, args)) for idv_label in labels[1:]]
    across_test_datasets = [RecFeatureDataset(info[0], info[1], info[2], is_train = False, is_fp16 = args.rec_fp16) for info in across_test_infos]
    across_test_vol_names = [info[3] for info in across_test_infos]
    across_test_dataloaders = [DataLoader(testset, batch_size = args.rec_batch_size) for testset in across_test_datasets]

    # ----- network --------------------
    model = RecFuseNetworkLinear(input_dim = (args.rec_knn_k * 3, args.rec_des_len * 4), output_dim = args.rec_len_embedding, num_ids = num_ids,
                                 channel_base = args.rec_channel_base, group_base = args.rec_group_base,
                                 dropout_ratio = 0.2, activation_method = "celu").cuda()
    model = model.half() if args.rec_fp16 else model
    if os.path.isfile(args.rec_model_load_path):
        model.load_state_dict(torch.load(args.rec_model_load_path, map_location = 'cuda:0')['network'])

    # model.eval()
    # results = OrderedDict()
    # results.update({name: [[], []] for name in trains[-1]})  # embedding, id
    # with torch.inference_mode():
    #     for idx, (names, feas, ids) in enumerate(train_dataloader):
    #         embeds, _ = model(feas.cuda(), mode = 1)
    #         for embed, name, _id, in zip(embeds.detach().cpu().numpy(), names[0], ids):
    #             if _id != processing_ids[-1]:
    #                 results[name][0].append(embed)
    #                 results[name][1].append(_id)
    #
    # for vol_name, (vol_pred, vol_label) in results.items():
    #     nd_pred, nd_label = np.array(vol_pred), np.array(vol_label)
    #     results[vol_name] = [nd_pred, nd_label]
    #
    # np.save("/home/cbmi/wyx/CenDer_PLOS_CompBio/notebooks/train", results)
    #
    # model.eval()
    # results = OrderedDict()
    # results.update({name: [[], []] for name in vals[-1]})  # embedding, id
    # with torch.inference_mode():
    #     for idx, (names, feas, ids) in enumerate(val_dataloader):
    #         embeds, _ = model(feas.cuda(), mode = 1)
    #         for embed, name, _id, in zip(embeds.detach().cpu().numpy(), names[0], ids):
    #             if _id != processing_ids[-1]:
    #                 results[name][0].append(embed)
    #                 results[name][1].append(_id)
    #
    # for vol_name, (vol_pred, vol_label) in results.items():
    #     nd_pred, nd_label = np.array(vol_pred), np.array(vol_label)
    #     results[vol_name] = [nd_pred, nd_label]
    #
    # np.save("/home/cbmi/wyx/CenDer_PLOS_CompBio/notebooks/val", results)
    #
    # model.eval()
    # results = OrderedDict()
    # results.update({name: [[], []] for name in tests[-1]})  # embedding, id
    # with torch.inference_mode():
    #     for idx, (names, feas, ids) in enumerate(within_test_dataloader):
    #         embeds, _ = model(feas.cuda(), mode = 1)
    #         for embed, name, _id, in zip(embeds.detach().cpu().numpy(), names[0], ids):
    #             if _id != processing_ids[-1]:
    #                 results[name][0].append(embed)
    #                 results[name][1].append(_id)
    #
    # for vol_name, (vol_pred, vol_label) in results.items():
    #     nd_pred, nd_label = np.array(vol_pred), np.array(vol_label)
    #     results[vol_name] = [nd_pred, nd_label]
    #
    # np.save("/home/cbmi/wyx/CenDer_PLOS_CompBio/notebooks/test", results)
    #
    # exit(-1)

    criterion = RecMarginalCosLossNetwork(len_embedding = args.rec_len_embedding, coefficients = args.rec_loss_coefficients, hypersphere_radius = args.rec_hypersphere_radius).cuda()
    criterion = criterion.half() if args.rec_fp16 else criterion

    train_val_procedure(model = model,
                        criterion = criterion,
                        train_dataloader = train_dataloader,
                        val_dataloader = val_dataloader,
                        within_test_dataloader = within_test_dataloader,
                        across_test_dataloaders = across_test_dataloaders,
                        across_test_vol_names = across_test_vol_names,
                        cfg = cfg,
                        test_vols = tests[3],
                        batch_size = args.rec_batch_size,
                        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3),
                        is_fp16 = args.rec_fp16,
                        num_ids = num_ids,
                        processing_ids = processing_ids,
                        hr = args.rec_hypersphere_radius,
                        num_epochs = args.rec_epoch,
                        checkpoint_timestamp = args.checkpoint_timestamp,
                        tb_writer = SummaryWriter(os.path.join(args.rec_tensorboard_root, args.checkpoint_timestamp)),
                        model_save_path = os.path.join(args.rec_model_save_path, f"rec_{args.checkpoint_timestamp}.ckpt"),
                        )

    # ==========  evaluate benchmarks  ====================
    from src.benchmarks.test2 import evaluate_benchmark
    from src.benchmarks.datasets.CeNDeR import Dataset_CeNDeR

    model.eval()
    test_batch_size = 32  # keep the same with fDNC

    # benchmark leifer 2017
    dataset = Dataset_CeNDeR(glob(os.path.join(args.data_root, "data/benchmarks/supp/e1/test_tracking", "*.npy")), is_fp16 = args.rec_fp16, num_pool = args.rec_num_workers)
    dataloader = DataLoader(dataset, args.rec_batch_size)
    accs = evaluate_benchmark(dataloader, model, refer_idx = 16, verbose = False, test_batch_size = test_batch_size)
    print_info_message(f"NeRVE: accuracy:  {accs[0] * 100:.2f} \t num_gts: {accs[2]:.2f} \t num_hits: {(accs[4]):.2f}")

    # benchmark NeuroPAL
    dataset = Dataset_CeNDeR(glob(os.path.join(args.data_root, "data/benchmarks/supp/e1/test_neuropal_our", "*.npy")), is_fp16 = args.rec_fp16, num_pool = args.rec_num_workers)
    dataloader = DataLoader(dataset, args.rec_batch_size)
    accs = np.array([evaluate_benchmark(dataloader, model, refer_idx = ref_idx, test_batch_size = test_batch_size) for ref_idx in range(dataset.num_vols)])
    print_info_message(f"NeuroPAL Yu: accuracy: {np.mean(accs[:, 0] * 100):.2f} ± {np.std(accs[:, 0] * 100):.2f} \t num_gts: {np.mean(accs[:, 2]):.2f} ± {np.std(accs[:, 2]):.2f} \t "
                       f"num_hits: {np.mean(accs[:, 4]):.2f} ± {np.std(accs[:, 4]):.2f}")

    # benchmark NeuroPAL Chaudhary
    dataset = Dataset_CeNDeR(glob(os.path.join(args.data_root, "data/benchmarks/supp/e1/test_neuropal_Chaudhary", "*.npy")), is_fp16 = args.rec_fp16, num_pool = args.rec_num_workers)
    dataloader = DataLoader(dataset, args.rec_batch_size)
    accs = np.array([evaluate_benchmark(dataloader, model, refer_idx = ref_idx, test_batch_size = test_batch_size) for ref_idx in range(dataset.num_vols)])
    print_info_message(f"NeuroPAL Chaudhary: accuracy: {np.mean(accs[:, 0] * 100):.2f} ± {np.std(accs[:, 0] * 100):.2f} \t num_gts: {np.mean(accs[:, 2]):.2f} ± {np.std(accs[:, 2]):.2f} \t "
                       f"num_hits: {np.mean(accs[:, 4]):.2f} ± {np.std(accs[:, 4]):.2f}")
