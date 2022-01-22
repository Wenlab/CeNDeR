# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

import os
from tqdm import tqdm
from typing import Dict
from multiprocessing import Pool

from recognition.inference.id_map import id_map
from recognition.inference.rec_dl_utils import nn_infer
from recognition.inference.rec_infer_utils import proposal_prediction
from recognition.inference.feature_maker.load_features import make_one_volume_neuronal_features


def rec_infer_run(merge_results: Dict, volumes: Dict, args, fea_len, others_class_start_id: int = 1000):
    features = rec_feature_build_mulitprocess(merge_results, volumes, args)
    raw_results = nn_infer(args, features, fea_len, id_map)
    pred_results = proposal_prediction(merge_results, raw_results, id_map, others_class_start_id)
    return pred_results


def rec_feature_build_mulitprocess(raw_neurons: Dict, volumes: Dict, args):
    params = [[name, raw_neurons[name], volumes[name][3], args] for name in volumes.keys()]
    with Pool(min(os.cpu_count() // 2, len(params))) as p:
        with tqdm(p.imap_unordered(build_rec_feature, params), total = len(params), desc = "S.4 recognition.feature_making") as pbar:
            features = {name: fea for name, fea in list(pbar)}
    return features


def build_rec_feature(params):
    name, volume_3d_result, ccords, args = params
    fea = {i: knn_fea + den_fea for i, [knn_fea, den_fea] in make_one_volume_neuronal_features(name, volume_3d_result, ccords, args).items()}
    return name, fea
