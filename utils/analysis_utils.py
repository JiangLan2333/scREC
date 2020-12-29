#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/29 上午9:46
# @Author  : Lan Jiang
# @File    : analysis_utils.py

import pickle
import numpy as np

import scanpy as sc
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import homogeneity_score


def run_louvain(sc_data, sc_labels, range_min=0, range_max=3, max_steps=30):
    adata = sc.AnnData(sc_data)
    adata.obs['label'] = sc_labels
    sc.pp.neighbors(adata, use_rep='X')
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=['label'])

    n_cluster = np.unique(sc_labels).shape[0]

    this_step = 0
    this_min = float(range_min)
    this_max = float(range_max)
    while this_step < max_steps:
        #         print('step ' + str(this_step))
        this_resolution = this_min + ((this_max - this_min) / 2)
        sc.tl.louvain(adata, resolution=this_resolution)
        this_clusters = adata.obs['louvain'].nunique()
        #         print('got ' + str(this_clusters) + ' at resolution ' + str(this_resolution))
        if this_clusters > n_cluster:
            this_max = this_resolution
        elif this_clusters < n_cluster:
            this_min = this_resolution
        else:
            #             return(this_resolution, adata)
            ari = adjusted_rand_score(adata.obs['label'], adata.obs['louvain'])
            ami = adjusted_mutual_info_score(adata.obs['label'], adata.obs['louvain'])
            homo = homogeneity_score(adata.obs['label'], adata.obs['louvain'])
            print('Louvain:\tARI: %.3f, AMI: %.3f, Homo: %.3f' % (ari, ami, homo))
            return (adata.obs['label'], adata.obs['louvain'], ari, ami, homo)
        this_step += 1
    print('Cannot find the number of clusters')
    ari = adjusted_rand_score(adata.obs['label'], adata.obs['louvain'])
    ami = adjusted_mutual_info_score(adata.obs['label'], adata.obs['louvain'])
    homo = homogeneity_score(adata.obs['label'], adata.obs['louvain'])
    print('Louvain:\tARI: %.3f, AMI: %.3f, Homo: %.3f' % (ari, ami, homo))
    return adata.obs['label'], adata.obs['louvain'], ari, ami, homo


def fetch_low_dimension_data(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    h = model.get_h()
    _, labels = model.get_data()
    return h, labels
