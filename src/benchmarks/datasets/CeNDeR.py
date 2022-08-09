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
from tqdm import tqdm
from typing import Dict, List
from multiprocessing import Pool
from dataclasses import dataclass
from torch.utils.data import Dataset

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))
from src.common_utils.prints import pad_num
from src.preprocessing.training.utils import extract_annos
from src.recognition.training.extract_training_data import extract_preprocessing_json
from src.recognition.inference.feature_maker.load_features import make_one_volume_neuronal_features
from src.recognition.inference.feature_maker.utils import normalize_volume_neurons, calc_volume_mean


# ----- Dataset  --------------------
def _load_volume_fea_vec(param):
    vol_idx, path = param
    name = path[:-4].split("/")[-1]
    data = np.load(path)
    y = data[:, 0].astype(np.int32)
    fea_vec = data[:, 1:].astype(np.float32)  # 163 = pt + fea_vec
    _len = data.shape[0]
    return vol_idx, name, fea_vec, y, _len


@dataclass
class Dataset_CeNDeR(Dataset):
    paths: List
    is_fp16: bool = True
    num_pool: int = 8

    def __post_init__(self):
        # load
        self.names, self.Xs, self.ys, self.lens = self.load_data(self.paths)
        self.pad_name = [name for name, _len in zip(self.names, self.lens) for _ in range(_len)]
        self.pad_lens = [_len for name, _len in zip(self.names, self.lens) for _ in range(_len)]
        self.Xs = np.vstack(self.Xs)
        self.ys = np.hstack(self.ys)
        # setup
        self.num_vols = len(self.names)
        self.num_pts_a_volume = np.max(self.lens)

    def __len__(self):
        return len(self.pad_name)

    def __getitem__(self, idx):
        # Without pt clouds
        X, y = self.Xs[idx][3:], self.ys[idx]
        name, _l = self.pad_name[idx], self.pad_lens[idx]
        return torch.HalfTensor(X) if self.is_fp16 else torch.FloatTensor(X), torch.LongTensor([y]), name, _l

    def load_data(self, paths):
        names, Xs, ys, lens = [[[] for _ in paths] for _ in range(4)]

        params = [[vol_idx, path] for vol_idx, path in enumerate(sorted(paths))]
        with Pool(min(self.num_pool, len(params))) as p:
            # with tqdm(p.imap_unordered(_load_volume_fea_vec, params), total = len(params), desc = "Loading data") as pbar:
            for vol_idx, name, X, y, _len in list(p.imap_unordered(_load_volume_fea_vec, params)):
                names[vol_idx], Xs[vol_idx], ys[vol_idx], lens[vol_idx] = name, X, y, _len

        return names, Xs, ys, lens


def _save_rec_feature(params):
    pt_mode, fea_mode, root, name, rotation, neurons, pis, id_map, is_fp16, ccord, rec_z_scale, rec_worm_diagonal_line, rec_knn_k, rec_des_len, ratio = params
    # change the unit to keep the same with fDNC
    mean_pt = calc_volume_mean(neurons)
    neurons = normalize_volume_neurons(neurons, mean_pt, ratio)
    y, X_dot, fea_vecs = make_one_volume_neuronal_features(None, neurons, ccords = [[(p[0] - mean_pt[0]) * ratio, (p[1] - mean_pt[1]) * ratio] for p in ccord], mode = fea_mode,
                                                           rec_z_scale = rec_z_scale, rec_worm_diagonal_line = rec_worm_diagonal_line, rec_knn_k = rec_knn_k, rec_des_len = rec_des_len)
    if len(pis) != 0:
        for i in range(len(y)):
            y[i] = -1 if y[i] not in pis else id_map[y[i]]

    if pt_mode:
        res = np.hstack((np.array(X_dot), y[:, np.newaxis]))
    else:
        res = np.hstack((y[:, np.newaxis], np.array(X_dot), fea_vecs))
    os.makedirs(root, exist_ok = True)
    path = os.path.join(root, f"{name[0]}_{pad_num(name[1], 4)}")
    np.save(path, res.astype(np.float16) if is_fp16 else res.astype(np.float32))


@dataclass
class FeaVectorGenerator_CeNDeR:
    pis: List
    infos: List
    id_map: Dict
    cfg_fea: Dict
    data_root: str
    save_root: str
    idv_names: List
    ccords_root: str
    num_pool: int = 8
    fea_mode: int = 0
    pt_mode: bool = False
    is_fp16: bool = False
    name_reg: str = r"[iI]ma?ge?_?[sS]t(?:ac)?k_?\d+_dk?\d+.*[wW]\d+_?Dt\d{6}"

    def __post_init__(self):
        self.names, self.labels, self.ccords = self.load_labels(self.idv_names, self.infos)

    def save_feature_vector(self, rotation = False):
        # self._save_rec_feature(root = os.path.join(self.root, get_checkpoint_timestamp()), rotation = rotation)
        self._save_rec_feature(root = self.save_root, rotation = rotation)

    def load_labels(self, idv_names, infos):
        vols_xymin, vols_ccords = extract_preprocessing_json(os.path.join(self.ccords_root, "*/*.json"))
        names, labels, ccords = list(), list(), list()
        for idv_name, idv_info in zip(idv_names, infos):
            idv_labels = list()
            for vol_idx, (file_name, idxes) in enumerate(idv_info):
                raw_label = extract_annos(os.path.join(self.data_root, file_name), idxes, self.name_reg)
                _label = [{i: [[p[1] - vols_xymin[k][0], p[2] - vols_xymin[k][1], p[3] - vols_xymin[k][0], p[4] - vols_xymin[k][1], p[0]] for p in pp]
                           for i, pp in vol.items()} for k, vol in raw_label.items()]
                idv_labels.extend(_label)
                ccords.extend(vols_ccords[k] for k in raw_label.keys())
            names.extend([idv_name, i] for i in range(len(idv_labels)))
            labels.extend(idv_labels)
        return names, labels, ccords

    def _save_rec_feature(self, root, rotation):
        params = [[self.pt_mode, self.fea_mode, root, name, rotation, neurons, self.pis, self.id_map, self.is_fp16, ccord, self.cfg_fea['rec_z_scale'], self.cfg_fea['rec_worm_diagonal_line'],
                   self.cfg_fea['rec_knn_k'], self.cfg_fea['rec_des_len'], self.cfg_fea.get("ratio", 1.0)] for (name, neurons, ccord) in zip(self.names, self.labels, self.ccords)]
        with Pool(min(self.num_pool, len(params))) as p:
            with tqdm(p.imap_unordered(_save_rec_feature, params), total = len(params), desc = "Building feature vector") as pbar:
                list(pbar)


if __name__ == '__main__':
    random.seed(1024)
    np.random.seed(1024)

    parser = argparse.ArgumentParser(description = "Generate data")
    parser.add_argument('--root', type = str, default = '', help = '')
    parser.add_argument('--mode', type = int, default = 0)
    args = parser.parse_args()

    from src.recognition.training.configs import e1 as cfg

    if args.mode == 0:
        for name, info in cfg.dataset['animals']['label'].items():
            FeaVectorGenerator_CeNDeR(data_root = os.path.join(args.root, cfg.dataset["animals"]["label_root"]),
                                      cfg_fea = cfg.method['fea_vecs_setup'],
                                      idv_names = [name],
                                      name_reg = cfg.dataset["animals"]["name_reg"],
                                      infos = [info],
                                      ccords_root = os.path.join(args.root, cfg.dataset["animals"]["ccords_root"]),
                                      save_root = os.path.join(args.root, "data/benchmarks/CeNDeR/base", name),
                                      num_pool = 16,
                                      is_fp16 = False,
                                      pis = [],
                                      id_map = {},
                                      pt_mode = False,
                                      fea_mode = 0,
                                      ).save_feature_vector()
    elif args.mode == 1:
        from src.recognition.training.configs import e2_wa_nerve as e2_cfg

        for name, info in cfg.dataset['animals']['label'].items():
            FeaVectorGenerator_CeNDeR(data_root = os.path.join(args.root, cfg.dataset["animals"]["label_root"]),
                                      cfg_fea = e2_cfg.method['fea_vecs_setup'],
                                      idv_names = [name],
                                      name_reg = cfg.dataset["animals"]["name_reg"],
                                      infos = [info],
                                      ccords_root = os.path.join(args.root, cfg.dataset["animals"]["ccords_root"]),
                                      save_root = os.path.join(args.root, "data/benchmarks/CeNDeR/feavec4region", name),
                                      num_pool = 16,
                                      is_fp16 = False,
                                      pis = [],
                                      id_map = {},
                                      pt_mode = False,
                                      fea_mode = 1,
                                      ).save_feature_vector()
    elif args.mode == 2:
        setup_4fDNC = {
            "rec_z_scale"           : 1.0,
            "rec_worm_diagonal_line": 1.414,
            "rec_knn_k"             : 25,
            "rec_des_len"           : 10,
            "unit"                  : 0.3,  # 1 pixel = 0.3 um
            "ratio"                 : 0.3 / 84  # fdnc 1 unit = 84 um
        }

        for name, info in cfg.dataset['animals']['label'].items():
            FeaVectorGenerator_CeNDeR(data_root = os.path.join(args.root, cfg.dataset["animals"]["label_root"]),
                                      cfg_fea = setup_4fDNC,
                                      idv_names = [name],
                                      name_reg = cfg.dataset["animals"]["name_reg"],
                                      infos = [info],
                                      ccords_root = os.path.join(args.root, cfg.dataset["animals"]["ccords_root"]),
                                      save_root = os.path.join(args.root, "data/benchmarks/CeNDeR/ptcloud4fdnc", name),
                                      num_pool = 16,
                                      is_fp16 = False,
                                      pis = [],
                                      id_map = {},
                                      pt_mode = True,
                                      ).save_feature_vector()
