# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from typing import Dict, List
from torch.optim import optimizer
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from common_utils.prints import print_info_message, print_log_message
from common_utils.metric.rec import top_k_accuracy_score, top_1_accuracy_score, top_k_accuracy
from recognition.training.extract_training_data import find_intersected_neuron_ids, neurons2data
from recognition.inference.rec_dl_utils import RecFeatureDataset
from recognition.inference.rec_dl_utils import RecFuseNetworkLinear, RecMarginalCosLossNetwork
from recognition.inference.rec_embed_utils import extract_embedding_wlabel, store_neuronal_embedding


def test_one_epoch(Xs, ys, sample_names, test_vols, num_ids, model, batch_size, is_fp16):
    """Return every volume result and all"""
    dataset = RecFeatureDataset(Xs, ys, sample_names, is_train = False, is_fp16 = is_fp16)
    dataloader = DataLoader(dataset, batch_size = batch_size, drop_last = False, shuffle = True, pin_memory = False, num_workers = 1)
    names, labels, preds = list(), list(), torch.zeros((len(dataset), num_ids), dtype = torch.float16 if is_fp16 else torch.float32).cuda()
    model.eval()
    with torch.no_grad():
        for idx, (ns, feas, ids) in enumerate(dataloader):
            preds[idx * batch_size: idx * batch_size + len(ids)] = model(feas.cuda(), mode = 2)
            labels.extend(ids.tolist())
            names.extend([vol_name for vol_name in ns[0]])
    preds, labels = np.array(preds.cpu()), np.array(labels)
    sum_result, top3sum_result = top_1_accuracy_score(labels, preds), top_k_accuracy(labels, preds, k = 3)
    vol_info = {vol_name: [[], []] for vol_name in test_vols}
    for name, pred, label in zip(names, preds, labels):
        vol_info[name][0].append(pred)
        vol_info[name][1].extend(label)
    vol_results = {vol_name: top_1_accuracy_score(np.array(vol_label), np.array(vol_pred)) for vol_name, (vol_pred, vol_label) in vol_info.items()}

    return vol_results, sum_result, top3sum_result


def train_val_procedure(model: nn.Module, criterion: nn.Module,
                        train_dataloader: DataLoader, val_dataloader: DataLoader,
                        test_info,
                        optimizer: optimizer.Optimizer,
                        model_save_path: str,
                        tb_writer: SummaryWriter,
                        num_ids: int,
                        mcd_save_root,
                        num_epochs: int = 200,
                        uncertainty_interval_epoch: int = 0,
                        num_mcd_forward_propagation: int = 10,
                        **kwargs):
    results = [dict(), dict()]
    best_epoch = 0
    best_accuracy = [0.0, 0.0]
    for epoch in range(num_epochs):
        # =============== train ===============
        model.train()
        losses = 0.0
        rec_preds, rec_labels = list(), list()
        for iter_idx, rs in enumerate(train_dataloader):
            # scheduler.step(epoch + iter_idx / len(train_dataloader))
            ns, feas, ids = rs[0], rs[1].cuda(), rs[2].squeeze().cuda()  # ([neuron_name], [engineered_feature_vector], [identity (label)])
            # calculating
            optimizer.zero_grad()
            embedding, cos_dis = model(feas, mode = 1)
            loss = criterion(cos_dis, ids)
            loss.backward()
            optimizer.step()
            # recording
            losses += loss.item()
            rec_preds.extend(cos_dis.cpu().tolist())
            rec_labels.extend(ids.cpu().tolist())

        # results[0][epoch] = [top_k_accuracy_score(np.array(rec_labels), np.array(rec_preds), k = k) for k in (1, 3)]
        results[0][epoch] = [top_1_accuracy_score(np.array(rec_labels), np.array(rec_preds))]
        # print_log_message(f'TRAIN Epoch: {epoch}/{num_epochs} \t '
        #                   f'Loss: {losses / len(train_dataloader):.2f} \t '
        #                   f'Top-1 accuracy: {(results[0][epoch][0] * 100.0):.2f} % \t '
        #                   f'Top-3 accuracy: {(results[0][epoch][1] * 100.0):.2f} %')

        # =============== verification ===============
        model.eval()
        with torch.no_grad():
            embs = list()  # recording embedding into tensorboard
            rec_preds, rec_labels = list(), list()
            for ns, feas, ids in val_dataloader:
                # calculating
                feas, ids = feas.cuda(), ids.squeeze()
                embedding, cos_dis = model(feas, mode = 1)
                # recording
                embs.append(embedding)
                rec_preds.extend(cos_dis.cpu().tolist())
                rec_labels.extend(ids.tolist())

        # results[1][epoch] = [top_k_accuracy_score(np.array(rec_labels), np.array(rec_preds), k = k) for k in (1, 3)]
        results[1][epoch] = [top_1_accuracy_score(np.array(rec_labels), np.array(rec_preds))]
        # print_log_message(f'EVALUATE Epoch: {epoch}/{num_epochs} \t '
        #                   f'Top-1 accuracy: {(results[1][epoch][0] * 100.0):.2f} % \t '
        #                   f'Top-3 accuracy: {(results[1][epoch][1] * 100.0):.2f} %')
        # pytorch tensorboard: https://pytorch.org/docs/stable/tensorboard.html
        tb_writer.add_scalar('train_loss', losses / len(train_dataloader), global_step = epoch)
        tb_writer.add_scalars('top1_accuracy', {"train": results[0][epoch][0] * 100.0, "val": results[1][epoch][0] * 100.0}, global_step = epoch)
        # tb_writer.add_scalars('top3_accuracy', {"train": results[0][epoch][1] * 100.0, "val": results[1][epoch][1] * 100.0}, global_step = epoch)

        if best_accuracy[0] <= (results[1][epoch][0]):
            best_epoch, best_accuracy = epoch, results[1][epoch]
            torch.save({'network': model.state_dict()}, model_save_path)
            # print_info_message(f"Model has been saved in {model_save_path} !")
            tb_writer.add_embedding(mat = F.normalize(torch.cat(embs, dim = 0), dim = 1) * kwargs['hr'], metadata = rec_labels, global_step = epoch, tag = "samples")
            tb_writer.add_embedding(mat = F.normalize(model.norm_linear.weight, dim = 1) * kwargs['hr'], metadata = kwargs['processing_ids'], global_step = epoch, tag = "centers")
            # test set
            _, sum_result, top3_result = test_one_epoch(test_info[0], test_info[1], test_info[2], test_info[3], num_ids, model, kwargs['batch_size'], kwargs['is_fp16'])
            tb_writer.add_scalars('test_accuracy', {"top1": sum_result * 100.0}, global_step = epoch)

    tb_writer.close()
    print_info_message(f"------------------------------------------- >>>>>>>>>>>>>>>>>>>>>>>>>>>>"
                       f"The best val epoch of {kwargs['checkpoint_timestamp']} model is {best_epoch} \t "
                       f"Top-1 accuracy: {(best_accuracy[0] * 100.0):.2f} % \t "
                       # f"Top-3 accuracy: {(best_accuracy[1] * 100.0):.2f} % \t "
                       )
    # test
    model.load_state_dict(torch.load(model_save_path, map_location = 'cuda:0')['network'])
    vol_results, sum_result, top3_result = test_one_epoch(test_info[0], test_info[1], test_info[2], test_info[3], num_ids, model, kwargs['batch_size'], kwargs['is_fp16'])
    print_info_message(f"Testset top1 accuracy: {(sum_result * 100.0):.2f} \t top3 accuracy: {(top3_result * 100.0):.2f}  \n \t {vol_results}")

    return pd.DataFrame(results[1])


def deep_learning_run(neuronal_feas: Dict, **kwargs):
    # ----- data processing --------------------
    trains, vals, tests, num_ids, id_map, processing_ids = neurons2data(neuronal_feas.copy(),
                                                                        dataset_names = kwargs['dataset_names'],
                                                                        include_others_class = kwargs['include_others_class'],
                                                                        mode = kwargs['mode'],
                                                                        given_ids = sorted(kwargs['processing_ids']),
                                                                        )

    # ----- dataloader --------------------
    train_dataset = RecFeatureDataset(Xs = trains[0], ys = trains[1], names = trains[2], is_train = True, is_fp16 = kwargs['is_fp16'])
    train_dataloader = DataLoader(train_dataset, batch_size = kwargs['batch_size'], drop_last = True, shuffle = True, pin_memory = False, num_workers = 1)
    val_dataset = RecFeatureDataset(Xs = vals[0], ys = vals[1], names = vals[2], is_train = False, is_fp16 = kwargs['is_fp16'])
    val_dataloader = DataLoader(val_dataset, batch_size = kwargs['batch_size'], drop_last = False, shuffle = True, pin_memory = False, num_workers = 1)

    # ----- network --------------------
    model = RecFuseNetworkLinear(input_dim = kwargs["fea_len"], output_dim = kwargs['len_embedding'], num_ids = num_ids,
                                 channel_base = kwargs['channel_base'], group_base = kwargs['group_base'],
                                 dropout_ratio = 0.2, activation_method = "celu").cuda()
    model = model.half() if kwargs['is_fp16'] else model
    if os.path.isfile(kwargs["model_load_path"]):
        model.load_state_dict(torch.load(kwargs["model_load_path"], map_location = 'cuda:0')['network'])

    criterion = RecMarginalCosLossNetwork(len_embedding = kwargs['len_embedding'], coefficients = kwargs['loss_coefficients'], hypersphere_radius = kwargs['hypersphere_radius']).cuda()
    criterion = criterion.half() if kwargs['is_fp16'] else criterion

    train_val_result = train_val_procedure(model = model,
                                           criterion = criterion,
                                           train_dataloader = train_dataloader,
                                           val_dataloader = val_dataloader,
                                           test_info = tests,
                                           batch_size = kwargs['batch_size'],
                                           optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3),
                                           is_fp16 = kwargs['is_fp16'],
                                           num_ids = num_ids,
                                           processing_ids = processing_ids,
                                           hr = kwargs['hypersphere_radius'],
                                           num_epochs = kwargs['num_epochs'],
                                           checkpoint_timestamp = kwargs['checkpoint_timestamp'],
                                           tb_writer = SummaryWriter(os.path.join(kwargs['tb_root'], kwargs['checkpoint_timestamp'])),
                                           model_save_path = os.path.join(kwargs['model_save_path'], f"rec_{kwargs['checkpoint_timestamp']}.ckpt"),
                                           uncertainty_interval_epoch = kwargs['uncertainty_interval_epoch'],
                                           mcd_save_root = kwargs['mcd_save_prefix'],
                                           num_mcd_forward_propagation = kwargs['num_mcd_forward_propagation'],
                                           )

    return train_val_result
