# -*- coding: utf-8 -*-
# 
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com


import os
import re
import scipy
import torch
import numpy as np
import pandas as pd
from glob import glob
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.cluster import KMeans  # Clustering algorithms: https://scikit-learn.org/stable/modules/clustering.html
from typing import Dict, Tuple, List
import scipy.cluster.hierarchy as sch
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import cosine_distances

# Dimensionality reducation algorithms:
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE

clustering_metrics = [
    metrics.homogeneity_score,
    metrics.completeness_score,
    metrics.v_measure_score,
    metrics.adjusted_rand_score,
    metrics.adjusted_mutual_info_score,
]


def load_data(root: str, timestamp: str, modes: List[str]):
    data = {m: {p.split("/")[-1][:-4]: np.load(p, allow_pickle = True) for p in sorted(glob(os.path.join(root, m, timestamp, "*.npy")))} for m in modes}
    result = {m: [0, list(), list(), list(), list(), list(), list(), list(), list()] for m in modes}
    for m, res in data.items():
        for vol_name, (sum_accuracy, top1_acc, top2_acc, top3_acc, label, pred, embed) in res.items():
            result[m][0] = sum_accuracy
            result[m][1].extend([vol_name for _ in range(len(label))])
            result[m][2].extend([i for i in range(len(label))])
            result[m][3].extend([top1_acc for _ in range(len(label))])
            result[m][4].extend([top2_acc for _ in range(len(label))])
            result[m][5].extend([top3_acc for _ in range(len(label))])
            result[m][6].extend([l for l in label])
            result[m][7].extend([p for p in pred])
            result[m][8].extend([e for e in embed])
    return result


def load_data2(root, modes, nums):
    results = {}
    for mode, num in zip(modes, nums):
        data = np.load(os.path.join(root, f"{mode}.npy"), allow_pickle = True).item()
        keys = np.random.choice(list(data.keys()), num)
        print(keys)
        res = [[], []]
        for key in keys:
            res[0].extend([np.array(e) for e in data[key][0]])
            res[1].extend([np.array(l) for l in data[key][1]])
        results[mode] = [np.array(r).squeeze().tolist() for r in res]
    return results


def load_cluster_center(network_path: str):
    centers = F.normalize(torch.load(network_path, map_location = 'cpu')['network']['norm_linear.weight'], dim = 1).numpy()
    return centers


def merge_neuron_class(labels, embeds, num_ids: int):
    assert len(labels) == len(embeds)
    coll = [list() for _ in range(num_ids)]
    [coll[l].append(e) for l, e in zip(labels, embeds)]
    return coll


def calc_mean_embedding(embeds):
    mean_embeds = [np.mean(e, axis = 0) for e in embeds]
    mean_embeds = [m / np.linalg.norm(m) for m in mean_embeds]
    return mean_embeds


def calc_cos_dis_mean_center(mean_embeds, centers):
    # cosine distance = 1 - cosine similarity
    cos_dis = [1 - np.dot(m, c) for m, c in zip(mean_embeds, centers)]
    return cos_dis


def calc_within_class_cosine_distance(centers, embeds):
    cos_dis = [[1 - np.dot(c, e) for e in embed] for c, embed in zip(centers, embeds)]
    return cos_dis


def calc_between_class_cosine_distance(centers):
    cos_dis = cosine_distances(centers).tolist()
    return cos_dis


def change_multi_distribution_to_dataFrame(raw, ids, bins, density = False):
    stand_bins = np.histogram_bin_edges([i for ii in raw for i in ii], bins = bins)
    result = [(np.histogram(r, bins = stand_bins, density = density)[0] / len(r)).tolist() for r in raw] \
        if density else [np.histogram(r, bins = stand_bins, density = density)[0].tolist() for r in raw]
    coord = [0.5 * (stand_bins[i] + stand_bins[i + 1]) for i in range(len(stand_bins) - 1)]
    df = pd.DataFrame(data = result, columns = coord, index = ids)
    return df


def draw_hist_to_excel(raw, bins, density = False):
    stand_bins = np.histogram_bin_edges(raw, bins = bins)
    result = (np.histogram(raw, bins = stand_bins)[0] / len(raw)).tolist() if density else np.histogram(raw, bins = stand_bins)[0].tolist()
    coord = [0.5 * (stand_bins[i] + stand_bins[i + 1]) for i in range(len(stand_bins) - 1)]
    mean = np.mean(raw)
    median = np.median(raw)
    std = np.std(raw)
    result_ratio = [r / sum(result) for r in result]
    return result, result_ratio, coord, mean, median, std


def store_csv(path, df: pd.DataFrame):
    df.to_csv(path)


def cluster_cos_dist_matrix(matrix):
    """https://www.kaggle.com/sgalella/correlation-heatmaps-with-hierarchical-clustering"""
    pairwise_distances = sch.distance.squareform(matrix)
    linkage = sch.linkage(pairwise_distances, method = 'complete')  #
    cluster_distance_threshold = 0.1
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, criterion = 'distance')
    idx = np.argsort(idx_to_cluster_array)
    return matrix[idx, :][:, idx]


def calc_pca_embedding_center(embedding, labels, centers, mean_embed, dim = 3):
    X_reduced = PCA(n_components = dim).fit_transform(np.array(embedding + centers + mean_embed, dtype = np.float32))
    embedding_reduced, center_reduced, mean_embed_reduced = X_reduced[: len(embedding)], X_reduced[len(embedding):len(embedding) + len(centers)], X_reduced[len(embedding) + len(centers):]
    return embedding_reduced, labels, center_reduced, mean_embed_reduced


def calc_tsne_embedding_center(embedding, labels, centers, mean_embed, dim = 3, n_iter = 4000):
    X_reduced = TSNE(n_components = dim, perplexity = 12, metric = "cosine", n_jobs = 12, n_iter = n_iter).fit_transform(np.array(embedding + centers + mean_embed, dtype = np.float32))
    embedding_reduced, center_reduced, mean_embed_reduced = X_reduced[: len(embedding)], X_reduced[len(embedding):len(embedding) + len(centers)], X_reduced[len(embedding) + len(centers):]
    return embedding_reduced, labels, center_reduced, mean_embed_reduced


if __name__ == '__main__':
    time_stamp = "2021_12_26_18_24_46"
    ids = 164
    result = load_data(root = "/home/cbmi/elegans-neuron-net/embeds", timestamp = time_stamp, modes = ["train", "val", "test"])
    centers = load_cluster_center("/home/cbmi/wyx/CenDer_PLOS_CompBio/models/released/cender_rec_model.ckpt")
    embeds = merge_neuron_class(result['train'][6], result['train'][8], centers.shape[0])
    mean_embeds = calc_mean_embedding(embeds)
    # figure 1
    mean_center_cos_dis = calc_cos_dis_mean_center(mean_embeds, centers)
    y, y_ratio, x, mean, median, std = draw_hist_to_excel(mean_center_cos_dis, 15, density = True)
    # see tracking.ipynb for visulization
    embedding_reduced, labels, center_reduced, mean_embed_reduced = calc_pca_embedding_center(result['train'][8], result['train'][6], centers.tolist(), mean_embeds, dim = 3)

    # figure 2
    within_cos_dis = calc_within_class_cosine_distance(centers, embeds)  # drawn by origin
    df = change_multi_distribution_to_dataFrame(raw = within_cos_dis, ids = list(range(ids)), bins = 10, density = True)
    store_csv(f"/home/cbmi/elegans-neuron-net/tb_debug/{time_stamp}.csv", df)

    # figure 3
    between_cos_dis = np.array(calc_between_class_cosine_distance(centers))
    rearange_matrix = cluster_cos_dist_matrix(np.array(between_cos_dis))
    # sns.heatmap(between_cos_dis, cmap = 'BuPu', vmin = np.sort(np.unique(between_cos_dis.flatten()))[1], vmax = np.max(between_cos_dis))
    sns.clustermap(between_cos_dis, method = "complete", cmap = 'Purples', vmin = np.sort(np.unique(between_cos_dis.flatten()))[1] - 0.20,
                   vmax = np.max(between_cos_dis), annot = False, figsize = (12, 12))
    plt.savefig(f"/home/cbmi/elegans-neuron-net/tb_debug/{time_stamp}_cos.eps")
    plt.show()
