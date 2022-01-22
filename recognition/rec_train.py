# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

import os
import sys
import time
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from itertools import product

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
from common_utils.dataset_building import split_multi_individuals_datasets
from common_utils.prints import print_info_message, get_checkpoint_timestamp
from preprocessing.training.utils import extract_annos
from recognition.rec_infer import rec_infer_run
from recognition.training.method import training_procedure
from recognition.training.extract_training_data import extract_preprocessing_json
from recognition.training.extract_training_data import find_intersected_neuron_ids, find_all_neuron_ids, count_neuron_frequency, select_ids, select_ids_from_multi_individuals
from recognition.inference.feature_maker.load_features import make_one_volume_neuronal_features

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--name-reg', type = str, default = r"[iI]ma?ge?_?[sS]t(?:ac)?k_?\d+_dk?\d+.*[wW]\d+_?Dt\d{6}")
    parser.add_argument('--load-preprocess-result-root', type = str, default = "")
    parser.add_argument('--random-seed', type = int, default = 1212)
    parser.add_argument('--checkpoint-timestamp', type = str, default = get_checkpoint_timestamp())
    parser.add_argument('--label-root', type = str, default = "")
    # neural points setting
    parser.add_argument('--rec-topk', default = 3, type = int)
    parser.add_argument('--rec-others-class', action = "store_true")
    parser.add_argument('--rec-xoy-unit', type = float, default = 0.6, help = "um/pixel")
    parser.add_argument('--rec-z-unit', type = float, default = 1.5, help = "um/pixel")
    parser.add_argument('--rec-worm-diagonal-line', type = float, default = 400.0)
    # knn feature
    parser.add_argument('--rec-knn-k', type = int, default = 40)
    # neural density feature
    parser.add_argument('--rec-des-len', type = int, default = 40)
    # neuron recognition (train)
    parser.add_argument('--rec-fp16', action = "store_true")
    parser.add_argument('--rec-epoch', default = 400, type = int)
    parser.add_argument('--rec-num-workers', default = 8, type = int)
    parser.add_argument('--rec-batch-size', default = 256, type = int)
    parser.add_argument('--rec-model-load-path', type = str, default = "")
    parser.add_argument('--rec-model-save-path', type = str, default = '')
    # embedding method
    parser.add_argument('--rec-channel-base', type = int, default = 32)
    parser.add_argument('--rec-group-base', type = int, default = 4)
    parser.add_argument('--rec-len-embedding', type = int, default = 32)
    parser.add_argument('--rec-hypersphere-radius', type = int, default = 8)
    parser.add_argument('--rec-loss-coefficients', type = float, nargs = "+", default = [1.0, 0.0, 0.15])
    # uncertainty
    parser.add_argument('--rec-uncertainty-interval-epoch', type = int, default = 0)
    parser.add_argument('--mcd-save-prefix', type = str, default = "")
    parser.add_argument('--rec-num-mcd-forward-propagation', type = int, default = 10, help = "Monte Carlo Dropout: the number of MC samples")
    # tensorboard
    parser.add_argument('--rec-tensorboard-root', type = str, default = "")

    args = parser.parse_args()
    args.rec_z_scale = args.rec_z_unit / args.rec_xoy_unit
    print_info_message(args)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    # ----- data preparation --------------------
    vols_xymin, vols_ccords = extract_preprocessing_json(os.path.join(args.load_preprocess_result_root, "*/*.json"))  # vol_name: [[xmin, ymin], mass_of_center, anterior_y, posterior_y, ventral_x, dorsal_x]

    w2_labels = extract_annos(os.path.join(args.label_root, "PR_v051_ImgStk002_dk001_{w2_Dt210513220213Rebuild}_{red_from1516to2515}Dt210824.mat"), list(range(1, 31, 1)), args.name_reg)
    w4_labels = extract_annos(os.path.join(args.label_root, "PR_v045_ImgStk001_dk001_{w4_Dt210513}_{Dt210824_new_PR_onDt211222}.mat"), list(range(30)), args.name_reg)
    w6_labels = extract_annos(os.path.join(args.label_root, "PR_v150_ImgStk001_dk001_{w6_Dt210514001057rebuild}_{red_from2966to3965}_Dt211119.mat"), list(range(50)), args.name_reg)
    w6_labels_p2 = extract_annos(os.path.join(args.label_root, "PR_v006_ImgStk001_dk002_{w6_Dt210514001057rebuild}_{red_from4966to5965}}_Dt211123modified.mat"), list(range(10)), args.name_reg)
    w2_labels, w4_labels, w6_labels, w6_labels_p2 = [{k: {i: [[p[1] - vols_xymin[k][0], p[2] - vols_xymin[k][1], p[3] - vols_xymin[k][0], p[4] - vols_xymin[k][1], p[0]] for p in pp]
                                                          for i, pp in vol.items()} for k, vol in labels.items()} for labels in (w2_labels, w4_labels, w6_labels, w6_labels_p2)]

    labels, dataset_names = split_multi_individuals_datasets(labels = [{**w6_labels, **w6_labels_p2}, w4_labels, w2_labels],
                                                             indexes = [[(0, 40), (40, 50), (50, 60)], [(0, 20), (20, 25), (25, 30)], [(0, 20), (20, 25), (25, 30)]],
                                                             shuffle_type = 0)
    processing_ids = select_ids_from_multi_individuals([{**w6_labels, **w6_labels_p2}, w4_labels, w2_labels], 2, [54, 27, 27])

    print_info_message(f"\n \t training vols: {dataset_names[0]} \n \t validation vols: {dataset_names[1]} \n \t test vols: {dataset_names[2]} ")
    # ----- samples making --------------------
    engineering_feature = {key: make_one_volume_neuronal_features(key, labels[key], vols_ccords[key], args) for key in tqdm(labels.keys(), desc = " rec feature building")}
    result = training_procedure(method = "deep_learning",
                                dataset_names = dataset_names,
                                neuronal_feas = engineering_feature,
                                mode = (0, 1),
                                is_fp16 = args.rec_fp16,
                                batch_size = args.rec_batch_size,
                                channel_base = args.rec_channel_base,
                                len_embedding = args.rec_len_embedding,
                                group_base = args.rec_group_base,
                                loss_coefficients = args.rec_loss_coefficients,
                                hypersphere_radius = args.rec_hypersphere_radius,
                                checkpoint_timestamp = args.checkpoint_timestamp,
                                tb_root = args.rec_tensorboard_root,
                                model_save_path = args.rec_model_save_path,
                                model_load_path = args.rec_model_load_path,
                                uncertainty_interval_epoch = args.rec_uncertainty_interval_epoch,
                                num_epochs = args.rec_epoch,
                                mcd_save_prefix = args.mcd_save_prefix,
                                num_mcd_forward_propagation = args.rec_num_mcd_forward_propagation,
                                include_others_class = args.rec_others_class,
                                fea_len = [args.rec_knn_k * 3, args.rec_des_len * 4],
                                processing_ids = processing_ids,
                                )
