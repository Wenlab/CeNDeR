# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

import os
import gc
import sys
import time
import argparse
from glob import glob
from tqdm import tqdm
from typing import Dict

sys.path.append(os.path.join(os.path.dirname(__file__)))
from common_utils.dataset_building import split_processing_streams
from common_utils.prints import print_info_message, print_log_message
from preprocessing.pp_infer import auto_preprocess
from detection.det_infer import det_infer_main
from detection.inference.local_peak import parse_volume_peaks_multiprocess
from merge.xgraph_alignment import xneuronalign_multiprocess
from recognition.rec_infer import rec_infer_run
from recognition.inference.rec_infer_utils import store_result_as_json

# Initialize parameters
parser = argparse.ArgumentParser(description = "CenDer pipeline")
# --------------------------------- Stack loading and Stage 1: Pre-processing ---------------------------------
parser.add_argument('--zrange', type = tuple, default = (0, 18))
parser.add_argument('--process-stack-root', type = str, default = '', help = '')
parser.add_argument('--load-preprocess-result-root', type = str, default = '')
parser.add_argument('--save-preprocess-result-root', type = str, default = '')
parser.add_argument('--name-reg', type = str, default = r"[iI]ma?ge?_?[sS]t(?:ac)?k_?\d+_dk?\d+.*[wW]\d+_?Dt\d{6}")
parser.add_argument('--preprocessing-pixel-threshold', type = float, default = 1.8)  # sum of mean + 3 * std of a volume
parser.add_argument('--preprocessing-mode', type = int, default = 3)
parser.add_argument('--only-preprocessing', action = "store_true")
parser.add_argument('--volume-window', type = int, default = 8, help = "number of processing volumes once")

# --------------------------------- Stage 2: Neuronal region detection ---------------------------------
# peak searching
parser.add_argument('--det-pixel-threshold', type = float, default = 1.8)  # the mean + 3 * std of a volume
# ANN
parser.add_argument('--det-fp16', action = "store_true")
parser.add_argument('--det-num-workers', type = int, default = 4)
parser.add_argument('--det-batch-size', type = int, default = 256)
parser.add_argument('--det-input-size', type = int, default = 41)
parser.add_argument('--det-model-load-path', type = str, default = '')
parser.add_argument('--det-anchors-size', type = int, nargs = "+", default = [9])
parser.add_argument('--det-patches-size', type = int, nargs = "+", default = [15, 31, 41, 81])
# NMS
parser.add_argument('--det-nms-iou-threshold', type = float, default = 0.20)
parser.add_argument('--det-label-iou-threshold', type = float, default = 0.40)
parser.add_argument('--det-number-threshold', type = float, default = 100000)

# --------------------------------- Stage 3: 3D merging ---------------------------------
parser.add_argument('--merge-iou-threshold', type = float, default = 0.05)
parser.add_argument('--merge-span-threshold', type = int, default = 1)

# --------------------------------- Stage 4: Neuron recognition ---------------------------------
# neural points setting
parser.add_argument('--rec-fp16', action = "store_true")
parser.add_argument('--rec-other-class', action = "store_false")
parser.add_argument('--rec-worm-diagonal-line', type = float, default = 400.0)
parser.add_argument('--rec-z-scale', type = float, default = 1.5 / 0.6, help = "um/pixel")
# Embedding Feature
parser.add_argument('--rec-knn-k', type = int, default = 40)
# neural density feature
parser.add_argument('--rec-des-len', type = int, default = 40)
# neuron recognition
parser.add_argument('--rec-channel-base', type = int, default = 32)
parser.add_argument('--rec-group-base', type = int, default = 4)
parser.add_argument('--rec-len-embedding', type = int, default = 56)
# model setup
parser.add_argument('--rec-batch-size', default = 256, type = int)
parser.add_argument('--rec-model-load-path', type = str, default = "")

# --------------------------------- Result saving ---------------------------------
parser.add_argument('--json-store-root', type = str, default = "")

args = parser.parse_args()
print_info_message(args)

args.json_store_root = os.path.join(args.json_store_root, time.strftime('%m-%d'))

for stream in tqdm(split_processing_streams(glob(os.path.join(args.process_stack_root, '*.mat')), max_mats_one_stream = args.volume_window)):
    print_log_message(f"Processing stream: {stream}")

    # Stack loading and Stage 1: Pre-processing
    results = auto_preprocess(mode = args.preprocessing_mode, paths = stream, args = args)
    if args.only_preprocessing:
        continue
    # Stage 2: Neuronal region detection
    vol_peaks: Dict = parse_volume_peaks_multiprocess(auto_preprocess_result = results, peak_threshold = args.preprocessing_pixel_threshold)
    det_outputs = det_infer_main(args = args, vol_peaks = vol_peaks)
    # Stage 3: 3D merging
    merge_outputs = xneuronalign_multiprocess(iou_threshold = args.merge_iou_threshold, span_threshold = args.merge_span_threshold, volumes = vol_peaks, volumes_2d_regions_result = det_outputs)
    # Stage 4: Neuron recognition
    rec_outputs = rec_infer_run(merge_results = merge_outputs, volumes = results, args = args, fea_len = (args.rec_knn_k * 3, args.rec_des_len * 4), others_class_start_id = 2000)
    # Result saving
    store_result_as_json(root = args.json_store_root, outputs = rec_outputs, results = results)

    # Memory releasing
    del results, vol_peaks, det_outputs, merge_outputs, rec_outputs
    gc.collect()
