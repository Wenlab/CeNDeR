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
from torch import nn
from typing import Dict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from src.common_utils.metric.det import calc_det_score_wlabel
from src.common_utils.metric.merge import calc_merge_score_wlabel
from src.common_utils.dataset_building import split_multi_individuals_datasets
from src.common_utils.prints import print_log_message, print_info_message, get_checkpoint_timestamp
from src.preprocessing.pp_infer import auto_preprocess
from src.preprocessing.training.utils import extract_annos
from src.detection.inference.dataset import InferringNeuronDataset
from src.detection.det_infer import nms_multiprocess, det_infer_main
from src.detection.training.det_train_utils import TrainingNeuronDataset, label2merge_format
from src.detection.inference.bbox_regression_utils import calc_predicted_bbox, filter_bboxes_by_area
from src.detection.inference.local_peak import parse_volume_peaks_multiprocess
from src.detection.inference.network import MFDetectNetworkModule41, wing_loss
from src.merge.xgraph_alignment import xneuronalign_multiprocess


def train_val_procedure(num_epochs: int,
                        checkpoint_timestamp: str,
                        model_save_path: str,
                        tb_writer: SummaryWriter,
                        training_iou_threshold: float,
                        val_f1score_threshold: float,
                        nms_threshold: float):
    scaler = torch.cuda.amp.GradScaler()
    best_f1_score, best_epoch = 0.0, 0
    for epoch in range(num_epochs):
        train_dataset = TrainingNeuronDataset(vol_peak = {name: vol_peaks[name] for name in dataset_names[0]},
                                              input_size = args.det_input_size,
                                              anchors_size = args.det_anchors_size,
                                              patches_size = args.det_patches_size,
                                              labels = {name: labels[name] for name in dataset_names[0]},
                                              ls = {name: results[name][2][3][:2] for name in dataset_names[0]},
                                              iou_thresh = args.det_label_iou_threshold)
        train_dataloader = DataLoader(train_dataset, batch_size = args.det_batch_size, drop_last = True, shuffle = True, pin_memory = False, num_workers = args.det_num_workers)

        # =========  train  =========
        network.train()
        score_loss_sum, regression_loss_sum = 0.0, 0.0
        for iter_idx, (mrfs, scores, deltas) in enumerate(train_dataloader):
            # scheduler.step(epoch + iter_idx / len(train_dataloader))
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                pred_score, pred_delta = network(mrfs.cuda())
                score_loss = criterion(pred_score, scores.cuda())
                regression_loss = wing_loss(pred_delta, deltas.cuda(), w = 0.3)
                loss = 0.6 * score_loss + 0.4 * regression_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            score_loss_sum, regression_loss_sum = score_loss_sum + score_loss.item(), regression_loss_sum + regression_loss.item()
        tb_writer.add_scalars("train_loss", {"score": score_loss_sum / len(train_dataloader), "regression": regression_loss_sum / len(train_dataloader)}, global_step = epoch)
        # print_log_message(f'TRAIN Epoch: {epoch}/{num_epochs} Score Loss: {(score_loss_sum / (iter_idx + 1)):.2f}, Wing Loss: {(regression_loss_sum / (iter_idx + 1)):.2f},')

        # =========  verification  =========
        network.eval()
        with torch.no_grad():
            cn, cs, cb = list(), list(), list()
            for names, peaks, shifts, mrfs in val_dataloader:
                pred_score, pred_delta = network(mrfs.cuda())
                pred_score, pred_delta = torch.sigmoid(pred_score.cpu()), pred_delta.cpu()
                pos_idxes = torch.nonzero(pred_score > training_iou_threshold)[:, 0]
                bboxes = [calc_predicted_bbox(p.tolist(), s.tolist(), d.tolist()) for p, s, d in zip(peaks[pos_idxes], shifts[pos_idxes], pred_delta[pos_idxes])]
                b_idxes = [i for i, b in enumerate(bboxes) if filter_bboxes_by_area(b)]
                pos_idxes = pos_idxes[b_idxes]
                # store
                cn.extend(names[p] for p in pos_idxes)
                cs.extend(float(s) for s in pred_score[pos_idxes])
                cb.extend(bboxes[i] for i in b_idxes)
        candidate_scores, candidate_bboxes = dict(), dict()
        [candidate_scores.update({n: candidate_scores.get(n, []) + [s]}) for n, s in zip(cn, cs)]
        [candidate_bboxes.update({n: candidate_bboxes.get(n, []) + [b]}) for n, b in zip(cn, cb)]

        if len(candidate_bboxes) > 0:
            f1_score = calc_det_score_wlabel(nms_multiprocess(candidate_scores, candidate_bboxes, nms_threshold, args.det_number_threshold), val_labels, val_ls, threshold = val_f1score_threshold)[2]
        else:
            f1_score = 0.0
        tb_writer.add_scalar("validation_f1_score", f1_score, global_step = epoch)
        # print_log_message(f'EVALUATE Epoch: {epoch}/{num_epochs}, f1 score: {f1_score}')

        if best_f1_score <= f1_score:
            best_epoch, best_f1_score = epoch, f1_score
            torch.save({'network': network.state_dict()}, model_save_path)
            # print_log_message(f'{model_save_path} done!')

    print_info_message(f"---------------------------------- >>>>>>>>>>>>>>>>> The best val epoch of {checkpoint_timestamp} model is {best_epoch} \t  F1 score: {(best_f1_score * 100.0):.2f} % \t ")

    return best_f1_score, best_epoch, model_save_path


if __name__ == '__main__':
    # ----------- Parameters -----------
    parser = argparse.ArgumentParser(description = "Detection training phase")
    # ----- Preprocessing --------------------
    parser.add_argument('--checkpoint-timestamp', type = str, default = get_checkpoint_timestamp())
    parser.add_argument('--det-tensorboard-root', type = str, default = "tb_log/det")
    parser.add_argument('--zrange', type = tuple, default = (0, 18))
    parser.add_argument('--process-stack-root', type = str, default = 'data/dataset/raw')
    parser.add_argument('--load-preprocess-result-root', type = str, default = 'data/dataset/proofreading')
    parser.add_argument('--save-preprocess-result-root', type = str, default = '')
    parser.add_argument('--name-reg', type = str, default = r"[iI]ma?ge?_?[sS]t(?:ac)?k_?\d+_dk?\d+.*[wW]\d+_?Dt\d{6}")
    parser.add_argument('--preprocessing-pixel-threshold', type = float, default = 1.8)  # the mean + 3 * std of a volume
    parser.add_argument('--preprocessing-mode', type = int, default = 3)
    parser.add_argument('--random-seed', type = int, default = 1212)
    parser.add_argument('--label-root', type = str, default = "data/dataset/label")

    # ----- Det parameters --------------------
    parser.add_argument('--det-fp16', action = "store_true")
    parser.add_argument('--det-input-size', type = int, default = 41)
    parser.add_argument('--det-anchors-size', type = int, nargs = "+", default = [9])
    parser.add_argument('--det-patches-size', type = int, nargs = "+", default = [15, 31, 41, 81])
    # ----- Training parameters --------------------
    parser.add_argument('--det-label-iou-threshold', type = float, default = 0.20)
    parser.add_argument('--det-training-score-iou-threshold', type = float, default = 0.40)
    parser.add_argument('--det-f1score-threshold', type = float, default = 0.30)
    parser.add_argument('--det-nms-iou-threshold', type = float, default = 0.20)
    parser.add_argument('--det-number-threshold', type = float, default = 100000)
    parser.add_argument('--det-model-load-path', type = str, default = "")
    parser.add_argument('--det-model-save-path', type = str, default = "models")
    parser.add_argument('--num-epochs', type = int, default = 205)
    parser.add_argument('--lr', type = float, default = 0.05)
    parser.add_argument('--momentum', default = 0.9, type = float)
    parser.add_argument('--weight-decay', default = 5e-2, type = float)
    parser.add_argument('--gamma', default = 0.1, type = float)
    parser.add_argument('--det-num-workers', default = 4, type = int)
    parser.add_argument('--det-batch-size', default = 256, type = int)
    parser.add_argument('--epoch-interval', default = 15, type = int)
    # ----- Merge parameters --------------------
    parser.add_argument('--merge-f1score-threshold', type = float, default = 0.30)

    args = parser.parse_args()
    print_info_message(args)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # ----- Data --------------------
    w2_labels = extract_annos(os.path.join(args.label_root, "PR_v051_ImgStk002_dk001_{w2_Dt210513220213Rebuild}_{red_from1516to2515}Dt210824.mat"), list(range(1, 31, 1)), args.name_reg)
    w4_labels = extract_annos(os.path.join(args.label_root, "PR_v045_ImgStk001_dk001_{w4_Dt210513}_{Dt210824_new_PR_onDt211222}.mat"), list(range(30)), args.name_reg)
    w6_labels_p1 = extract_annos(os.path.join(args.label_root, "PR_v150_ImgStk001_dk001_{w6_Dt210514001057rebuild}_{red_from2966to3965}_Dt211119.mat"), list(range(50)), args.name_reg)
    w6_labels_p2 = extract_annos(os.path.join(args.label_root, "PR_v012_ImgStk001_dk002_{w6_Dt210514001057rebuild}_{red_from4966to5965}.mat"), list(range(50)), args.name_reg)
    w6_labels_p3 = extract_annos(os.path.join(args.label_root, "PR_v011_ImgStk003_dk001_{w6_Dt210514001057rebuild}_{red_from6966to7965}.mat"), list(range(50)), args.name_reg)
    w6_labels_p4 = extract_annos(os.path.join(args.label_root, "PR_v008_ImgStk004_dk001_{w6_Dt210514001057rebuild}_{red_from7966to8965}.mat"), list(range(50)), args.name_reg)
    w6_labels_p5 = extract_annos(os.path.join(args.label_root, "PR_v008_ImgStk005_dk001_{w6_Dt210514001057rebuild}_{red_from8966to9965}.mat"), list(range(50)), args.name_reg)
    w6_labels_p6 = extract_annos(os.path.join(args.label_root, "PR_v010_ImgStk007_dk001_{w6_Dt210514001057rebuild}_{red_from10966to11965}.mat"), list(range(50)), args.name_reg)
    w6_labels_p7 = extract_annos(os.path.join(args.label_root, "PR_v013_ImgStk009_dk001_{w6_Dt210514001057rebuild}_{red_from12966to13965}.mat"), list(range(50)), args.name_reg)

    mat_paths = [
        os.path.join(args.process_stack_root, "ImgStk002_dk001_{w2_Dt210513220213Rebuild}_{red_from1516to2515}.mat"),
        os.path.join(args.process_stack_root, "ImgStk001_dk001_{w4_Dt210513230327rebuild}_{red_from566to1565}.mat"),
        os.path.join(args.process_stack_root, "ImgStk001_dk001_{w6_Dt210514001057rebuild}_{red_from2966to3965}.mat"),
        os.path.join(args.process_stack_root, "ImgStk001_dk002_{w6_Dt210514001057rebuild}_{red_from4966to5965}.mat"),
        os.path.join(args.process_stack_root, "ImgStk003_dk001_{w6_Dt210514001057rebuild}_{red_from6966to7965}.mat"),
        os.path.join(args.process_stack_root, "ImgStk004_dk001_{w6_Dt210514001057rebuild}_{red_from7966to8965}.mat"),
        os.path.join(args.process_stack_root, "ImgStk005_dk001_{w6_Dt210514001057rebuild}_{red_from8966to9965}.mat"),
        os.path.join(args.process_stack_root, "ImgStk007_dk001_{w6_Dt210514001057rebuild}_{red_from10966to11965}.mat"),
        os.path.join(args.process_stack_root, "ImgStk009_dk001_{w6_Dt210514001057rebuild}_{red_from12966to13965}.mat"),
    ]
    w6_labels = {**w6_labels_p1, **w6_labels_p2, **w6_labels_p3, **w6_labels_p4, **w6_labels_p5, **w6_labels_p6, **w6_labels_p7}
    labels, dataset_names = split_multi_individuals_datasets(labels = [w6_labels, w4_labels, w2_labels],
                                                             indexes = [[(0, 30), (30, 50), (50, 350)], [(0, 0), (0, 0), (0, 30)], [(0, 0), (0, 0), (0, 30)]],
                                                             shuffle_type = 1)
    # print_info_message(f"\n \t training vols: {dataset_names[0]} \n \t validation vols: {dataset_names[1]} \n \t test vols: {dataset_names[2]} ")

    results = auto_preprocess(mode = args.preprocessing_mode, paths = mat_paths, args = args)
    results = {k: results[k] for k in list(labels.keys())}
    vol_peaks: Dict = parse_volume_peaks_multiprocess(results, args.preprocessing_pixel_threshold)

    val_dataset = InferringNeuronDataset(vol_peak = {name: vol_peaks[name] for name in dataset_names[1]},
                                         input_size = args.det_input_size,
                                         anchors_size = args.det_anchors_size,
                                         patches_size = args.det_patches_size,
                                         num_workers = args.det_num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size = args.det_batch_size, drop_last = False, shuffle = True, pin_memory = False, num_workers = args.det_num_workers)
    val_labels = {name: labels[name] for name in dataset_names[1]}
    val_ls = {name: results[name][2][3][:2] for name in dataset_names[1]}

    # ----------- Network & Loss Function-----------
    network = MFDetectNetworkModule41(num_channels = len(args.det_patches_size) + 1).cuda()
    if os.path.isfile(args.det_model_load_path):
        network.load_state_dict(torch.load(args.det_model_load_path, map_location = 'cuda:0')['network'])
    criterion = nn.BCEWithLogitsLoss().cuda()

    # ----------- Optimizer & Learning Schedule -----------
    # optimizer = torch.optim.Adam(network.parameters(), lr = args.lr)
    optimizer = torch.optim.SGD(network.parameters(), lr = args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5, 100)

    # ----------- Training part -----------
    if not os.path.isfile(args.det_model_load_path):
        best_f1_score, best_epoch, model_save_path = train_val_procedure(num_epochs = args.num_epochs,
                                                                         checkpoint_timestamp = args.checkpoint_timestamp,
                                                                         model_save_path = os.path.join(args.det_model_save_path, f"det_{args.checkpoint_timestamp}.ckpt"),
                                                                         tb_writer = SummaryWriter(os.path.join(args.det_tensorboard_root, args.checkpoint_timestamp)),
                                                                         training_iou_threshold = args.det_training_score_iou_threshold,
                                                                         val_f1score_threshold = args.det_f1score_threshold,
                                                                         nms_threshold = args.det_nms_iou_threshold, )
        args.det_model_load_path = model_save_path

    # ----------- Test -----------
    # ----------- Within -----------
    within_names = dataset_names[2]
    within_sp = {name: results[name][2][3][:2] for name in within_names}
    test_merge_results = xneuronalign_multiprocess(0.05, 1, {name: vol_peaks[name] for name in within_names}.copy(), label2merge_format({name: labels[name] for name in within_names}, within_sp))
    test_merge_f1_score = calc_merge_score_wlabel(test_merge_results, {name: labels[name] for name in within_names}.copy(),
                                                  sp = within_sp, threshold = args.merge_f1score_threshold, method = "F1SCORE")
    print_info_message(f"Test Merge Within scores (labels) (precision, recall, f1 score): {test_merge_f1_score[0:3]}")

    test_det_within_results = det_infer_main(args, {name: vol_peaks[name] for name in within_names})
    test_f1_score = calc_det_score_wlabel(test_det_within_results, labels = {name: labels[name] for name in within_names},
                                          sp = within_sp, threshold = args.det_f1score_threshold)
    print_info_message(f"Test Detection Within scores (precision, recall, f1 score): {test_f1_score[0:3]}")

    test_merge_results = xneuronalign_multiprocess(0.05, 1, {name: vol_peaks[name] for name in within_names}.copy(), test_det_within_results)
    test_merge_f1_score = calc_merge_score_wlabel(test_merge_results, {name: labels[name] for name in within_names}.copy(),
                                                  sp = within_sp, threshold = args.merge_f1score_threshold, method = "F1SCORE")
    print_info_message(f"Test Merge Within scores (prediction) (precision, recall, f1 score): {test_merge_f1_score[0]}")

    # ----------- Across -----------
    across_names = list(w2_labels.keys()) + list(w4_labels.keys())
    across_sp = {name: results[name][2][3][:2] for name in across_names}

    test_merge_results = xneuronalign_multiprocess(0.05, 1, {name: vol_peaks[name] for name in across_names}.copy(), label2merge_format({name: labels[name] for name in across_names}, across_sp))
    test_merge_f1_score = calc_merge_score_wlabel(test_merge_results, {name: labels[name] for name in across_names}.copy(),
                                                  sp = {name: results[name][2][3][:2] for name in across_names}, threshold = args.merge_f1score_threshold, method = "F1SCORE")
    print_info_message(f"Test Merge Within scores (labels) (precision, recall, f1 score): {test_merge_f1_score[0:3]}")

    test_det_across_results = det_infer_main(args, {name: vol_peaks[name] for name in across_names})
    test_f1_score = calc_det_score_wlabel(test_det_across_results, labels = {name: labels[name] for name in across_names},
                                          sp = across_sp, threshold = args.det_f1score_threshold)
    print_info_message(f"Test Detection Across scores (precision, recall, f1 score): {test_f1_score[0:3]}")

    test_merge_results = xneuronalign_multiprocess(0.05, 1, {name: vol_peaks[name] for name in across_names}.copy(), test_det_across_results)
    test_merge_f1_score = calc_merge_score_wlabel(test_merge_results, {name: labels[name] for name in across_names}.copy(),
                                                  sp = across_sp, threshold = args.merge_f1score_threshold, method = "F1SCORE")
    print_info_message(f"Test Merge Across scores (prediction) (precision, recall, f1 score): {test_merge_f1_score[0]}")
