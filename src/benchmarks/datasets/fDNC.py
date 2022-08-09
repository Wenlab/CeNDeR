import os
import sys
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
from typing import Dict, List
from collections import Counter
from multiprocessing import Pool
from dataclasses import dataclass
from torch.utils.data import Dataset

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))
from src.common_utils.prints import pad_num
from src.recognition.inference.feature_maker.load_features import make_one_volume_neuronal_features_4ptcloud


def count_frequecy(ys, num_vols):
    counter = [(k, v) for k, v in sorted(Counter(ys).items(), key = lambda item: item[0])]  # sorted by id number
    counter = [(k, v) for k, v in sorted(counter, key = lambda item: item[1], reverse = True)]
    unique_ids, frequencies = [v[0] for v in counter], [v[1] for v in counter]
    ratios = [fr / num_vols for fr in frequencies]
    return unique_ids, frequencies, ratios


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
class Dataset_fDNC(Dataset):
    paths: List
    is_fp16: bool = True
    num_pool: int = 8

    def __post_init__(self):
        # load
        self.names, self.Xs, self.ys, self.lens = self.load_data(self.paths)
        # setup
        self.num_vols = len(self.names)
        self.num_pts_a_volume = np.max(self.lens)

    def __len__(self):
        return self.num_vols

    def __getitem__(self, idx):
        X, y, _len = self.Xs[idx], self.ys[idx], self.lens[idx]
        return torch.HalfTensor(X) if self.is_fp16 else torch.FloatTensor(X), torch.LongTensor(y), torch.LongTensor([_len])

    def load_data(self, paths):
        names, Xs, ys, lens = [[[] for _ in paths] for _ in range(4)]

        params = [[vol_idx, path] for vol_idx, path in enumerate(sorted(paths))]
        with Pool(min(self.num_pool, len(params))) as p:
            # with tqdm(p.imap_unordered(_load_volume_fea_vec, params), total = len(params), desc = "Loading data") as pbar:
            for vol_idx, name, X, y, _len in list(p.imap_unordered(_load_volume_fea_vec, params)):
                names[vol_idx], Xs[vol_idx], ys[vol_idx], lens[vol_idx] = name, X, y, _len

        return names, Xs, ys, lens


def collate_fn_volumes(data):
    lens = torch.LongTensor([item[2] for item in data])
    batch_size = len(data)
    max_lens = torch.max(lens)
    batch_X = torch.zeros((batch_size, max_lens, data[0][0].shape[1]), dtype = torch.float32)
    batch_y = torch.zeros((batch_size, max_lens), dtype = torch.long)
    for i, (X, y, _len) in enumerate(data):
        batch_X[i, :_len] = X
        batch_y[i, :_len] = y
    return torch.FloatTensor(batch_X), torch.LongTensor(batch_y), lens


# ----- Generate Feature vector  --------------------
def _pad_name(name: str):
    name = "_".join([pad_num(i, 4) if i.isdigit() and len(i) <= 4 else i for i in name.split("_")])
    return name


def _load_pt_cloud(param):
    vol_idx, path = param
    name = path[:-4].split("/")[-1]
    data = np.load(path)
    X = data[:, :3].astype(np.float32)
    y = data[:, 3].astype(np.int32)
    _len = data.shape[0]
    return vol_idx, name, X, y, _len


def _rotate_pt_cloud(pt_cloud):
    theta = np.random.rand(1)[0] * 2 * np.pi
    r_m = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
    pt_cloud[:, :2] = np.matmul(pt_cloud[:, :2], r_m)
    return pt_cloud


def _save_rec_feature(params):
    root, name, rotate, X, y, is_fp16, ccords_root, rec_z_scale, rec_worm_diagonal_line, rec_knn_k, rec_des_len, ratio = params
    if rotate:
        X = _rotate_pt_cloud(X)
    ccord = _extract_ccords_fdnc(ccords_root, name)
    if ratio != 1:
        X = X * ratio
        ccord = [[p[0] * ratio, p[1] * ratio] for p in ccord]
    X_dot, fea = make_one_volume_neuronal_features_4ptcloud(X, ccord, rec_z_scale = rec_z_scale, rec_worm_diagonal_line = rec_worm_diagonal_line, rec_knn_k = rec_knn_k, rec_des_len = rec_des_len)
    res = np.hstack((y[:, np.newaxis], X, fea))
    # res = np.hstack((y[:, np.newaxis], X_dot, fea))  # TODO
    name = _pad_name(name)
    root = root if "synthetic" not in root else os.path.join(root, "_".join(name.split("_")[:2]))
    os.makedirs(root, exist_ok = True)
    print(os.path.join(root, name))
    np.save(os.path.join(root, name), res.astype(np.float16) if is_fp16 else res.astype(np.float32))


def _extract_ccords_fdnc(root: str, name: str):
    path = os.path.join(root, f"{name}.json")
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as file:
        json_file = json.load(file)["shapes"]
        for item in json_file:
            if item['label'] == 'anterior_y':
                anterior_y = [v for v in item['points'][0]]
            if item['label'] == 'posterior_y':
                posterior_y = [v for v in item['points'][0]]
            if item['label'] == 'ventral_x':
                ventral_x = [v for v in item['points'][0]]
            if item['label'] == 'dorsal_x':
                dorsal_x = [v for v in item['points'][0]]
            if item['label'] == 'mass_of_center':
                mass_of_center = [v for v in item['points'][0]]
    ccords = [mass_of_center, anterior_y, posterior_y, ventral_x, dorsal_x]
    ccords = [[(p - 200) / 400 for p in pt] for pt in ccords]
    return ccords


@dataclass
class FeaVectorGenerator_fDNC:
    paths: List
    cfg_fea: Dict
    save_root: str = ""
    ccords_root: str = ""
    is_fp16: bool = False
    num_pool: int = 8

    def __post_init__(self):
        self.names, self.Xs, self.ys = self.load_pt_clouds(self.paths)

    def save_feature_vector(self, rotation = False):
        # self._save_rec_feature(root = os.path.join(self.root, get_checkpoint_timestamp()), rotation = rotation)
        self._save_rec_feature(root = self.save_root, rotation = rotation)

    # @staticmethod
    def load_pt_clouds(self, paths):
        names, Xs, ys, lens = [[[] for _ in paths] for _ in range(4)]

        params = [[vol_idx, path] for vol_idx, path in enumerate(paths)]
        with Pool(min(self.num_pool, len(params))) as p:
            with tqdm(p.imap_unordered(_load_pt_cloud, params), total = len(params), desc = "Loading point clouds") as pbar:
                for vol_idx, name, X, y, _len in list(pbar):
                    names[vol_idx], Xs[vol_idx], ys[vol_idx], lens[vol_idx] = name, X, y, _len

        return names, Xs, ys

    def _save_rec_feature(self, root, rotation):
        params = [[root, name, rotation, X, y, self.is_fp16, self.ccords_root, self.cfg_fea['rec_z_scale'], self.cfg_fea['rec_worm_diagonal_line'],
                   self.cfg_fea['rec_knn_k'], self.cfg_fea['rec_des_len'], self.cfg_fea.get("ratio", 1.0)] for (name, X, y) in zip(self.names, self.Xs, self.ys)]
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

    if args.mode == 0:
        from src.recognition.training.configs import e1 as e1_cfg

        for cfg in (e1_cfg.benchmark_NeRVE, e1_cfg.benchmark_NeuroPAL, e1_cfg.benchmark_NeuroPAL_Chaudhary):
            FeaVectorGenerator_fDNC(paths = glob(os.path.join(args.root, cfg['animals']['root'], "**/*.npy")),
                                    ccords_root = os.path.join(args.root, cfg['animals']['preprocessing']),
                                    save_root = os.path.join(args.root, cfg['animals']['fea_vecs']),
                                    num_pool = 16,
                                    is_fp16 = False,
                                    cfg_fea = e1_cfg.method['fea_ves_setup_fDNC4CeNDeR'],
                                    ).save_feature_vector()

    elif args.mode == 1:
        # (e2_wa_nerve) NeRVE within-animal experiment
        from src.recognition.training.configs import e2_wa_nerve as e2_cfg

        FeaVectorGenerator_fDNC(paths = glob(os.path.join(args.root, e2_cfg.dataset['animals']['root']['raw'], "**/*.npy")),
                                ccords_root = os.path.join(args.root, e2_cfg.dataset['animals']['root']['preprocessing']),
                                save_root = os.path.join(args.root, e2_cfg.dataset['animals']['root']['fea_vecs']),
                                num_pool = 16,
                                is_fp16 = False,
                                cfg_fea = e2_cfg.method['fea_ves_setup_fDNC4CeNDeR'],
                                ).save_feature_vector(rotation = False)

        for cfg in (e2_cfg.benchmark_NeuroPAL, e2_cfg.benchmark_NeuroPAL_Chaudhary):
            FeaVectorGenerator_fDNC(paths = glob(os.path.join(args.root, cfg['animals']['root'], "**/*.npy")),
                                    ccords_root = os.path.join(args.root, cfg['animals']['preprocessing']),
                                    save_root = os.path.join(args.root, cfg['animals']['fea_vecs']),
                                    num_pool = 16,
                                    is_fp16 = False,
                                    cfg_fea = e2_cfg.method['fea_ves_setup_fDNC4CeNDeR'],
                                    ).save_feature_vector()
