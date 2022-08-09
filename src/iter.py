# -*- coding: utf-8 -*-
# 
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

import os
import gc

from src.common_utils.prints import print_log_message, print_warning_message
from src.preprocessing.pp_infer import auto_preprocess
from src.detection.det_infer import det_infer_main
from src.detection.inference.local_peak import parse_volume_peaks_multiprocess
from src.merge.xgraph_alignment import xneuronalign_multiprocess
from src.recognition.rec_infer import tracking_infer_run
from src.recognition.inference.rec_infer_utils import store_result_as_json


def infer_one_batch(args, stream, error_file_path: str, is_saved: bool = True, others_class_start_id: int = 2000):
    print_log_message(f"Processing stream: {stream}")
    try:
        # Stack loading and Stage 1: Pre-processing
        preprc_results = auto_preprocess(mode = args.preprocessing_mode, paths = stream, args = args)
        if args.only_preprocessing:
            return
        # Stage 2: Neuronal region detection
        vol_peaks = parse_volume_peaks_multiprocess(auto_preprocess_result = preprc_results, peak_threshold = args.preprocessing_pixel_threshold)
        det_outputs = det_infer_main(args = args, vol_peaks = vol_peaks)
        # Stage 3: 3D merging
        merge_outputs = xneuronalign_multiprocess(iou_threshold = args.merge_iou_threshold, span_threshold = args.merge_span_threshold, volumes = vol_peaks, volumes_2d_regions_result = det_outputs)
        # Stage 4: Neuron recognition
        rec_outputs = tracking_infer_run(merge_results = merge_outputs, volumes = preprc_results, args = args, fea_len = (args.rec_knn_k * 3, args.rec_des_len * 4), others_class_start_id = others_class_start_id)

        # Result saving
        if is_saved:
            store_result_as_json(root = args.json_store_root, outputs = rec_outputs, preprc_results = preprc_results)
        return rec_outputs, preprc_results

    except Exception as e:
        print_warning_message(f" --->     CenDer can't process {stream}. \t Because {e}")
        os.makedirs(args.json_store_root, exist_ok = True)
        with open(error_file_path, 'a') as error_file:
            for stack_path in stream:
                error_file.write(stack_path + '\n')
