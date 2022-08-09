# -*- coding: utf-8 -*-
# 
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com


import torch
from typing import Dict
from torch.utils.data import Dataset


class RecFeatureDataset(Dataset):
    def __init__(self,
                 Xs, ys, names,
                 is_train: bool = True,
                 is_fp16: bool = False):
        self.Xs = Xs
        self.ys = ys
        self.names = names
        self.is_train = is_train
        self.is_fp16 = is_fp16

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        X, y = self.Xs[idx], self.ys[idx]
        name = self.names[idx]

        return name, torch.HalfTensor(X) if self.is_fp16 else torch.FloatTensor(X), torch.LongTensor([y])


class InferRecFeatureDataset(Dataset):
    def __init__(self, vols_neurons_feas: Dict,
                 is_fp16: bool = False):
        self.is_fp16 = is_fp16

        self.feas, self.neuron_id = list(), list()
        for vol_name, vol_neurons_feas in vols_neurons_feas.items():
            for neuron_id, fea in vol_neurons_feas.items():
                self.feas.append(fea)
                self.neuron_id.append([vol_name, neuron_id])

    def __len__(self):
        return len(self.neuron_id)

    def __getitem__(self, idx):
        return self.neuron_id[idx], torch.HalfTensor(self.feas[idx]) if self.is_fp16 else torch.FloatTensor(self.feas[idx])