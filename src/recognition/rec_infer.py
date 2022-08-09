# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple
from multiprocessing import Pool

from src.recognition.inference.network import RecFuseNetworkLinear
from src.recognition.inference.dataset import InferRecFeatureDataset
from src.recognition.inference.rec_infer_utils import proposal_prediction
from src.recognition.inference.feature_maker.load_features import make_one_volume_neuronal_features


def tracking_infer_run(merge_results: Dict, volumes: Dict, args, fea_len, others_class_start_id: int = 1000):
    features = rec_feature_build_mulitprocess(merge_results, volumes, args)
    raw_results = nn_infer(args, features, fea_len)
    pred_results = proposal_prediction(merge_results, raw_results, os.path.join(args.json_store_root, "template.npy"), others_class_start_id)
    return pred_results


def rec_feature_build_mulitprocess(raw_neurons: Dict, volumes: Dict, args):
    params = [[name, raw_neurons[name], volumes[name][3], args] for name in volumes.keys()]
    with Pool(min(os.cpu_count() // 2, len(params))) as p:
        with tqdm(p.imap_unordered(build_rec_feature, params), total = len(params), desc = "S.4 recognition.feature_making") as pbar:
            features = {name: fea for name, fea in list(pbar)}
    return features


def build_rec_feature(params):
    name, volume_3d_result, ccords, args = params
    ids, _, feas = make_one_volume_neuronal_features(name, volume_3d_result, ccords, args = args)
    feas = {i: fea for i, fea in zip(ids, feas)}
    return name, feas


# ----- Infer main procedure --------------------
def nn_infer(args, vols_neurons_feature: Dict, fea_len: Tuple):
    # ----------- Dataloader -----------
    dataset = InferRecFeatureDataset(vols_neurons_feature, is_fp16 = args.rec_fp16)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 256, drop_last = False, shuffle = False, pin_memory = False, num_workers = 4)
    # ----------- Network -----------
    network = RecFuseNetworkLinear(input_dim = fea_len, output_dim = args.rec_len_embedding, num_ids = 164, channel_base = args.rec_channel_base,
                                   group_base = args.rec_group_base, dropout_ratio = 0.2, activation_method = "celu").cuda()
    network.load_state_dict(torch.load(args.rec_model_load_path, map_location = 'cuda:0')['network'])
    network = network.half() if args.rec_fp16 else network
    # ----------- Main Procedure -----------
    results = neural_network(dataloader, network, {key: [[], []] for key in vols_neurons_feature.keys()})  # [merged_id, prob]
    return results


def neural_network(dataloader, network, results):
    network.eval()
    with torch.inference_mode():
        for i, (n_id, fea) in enumerate(tqdm(dataloader, desc = "S.4 recognition.ANN")):
            embeds, _ = network(fea.cuda(), mode = 1)
            for embed, name, _id in zip(embeds.detach().cpu().numpy(), *n_id):
                results[name][0].append(int(_id))
                results[name][1].append(embed)

    return {vol_name: [merged_ids, np.array(embeds)] for vol_name, (merged_ids, embeds) in results.items()}
