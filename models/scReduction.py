#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/28 上午10:40
# @Author  : Lan Jiang
# @File    : scReduction.py

import numpy as np
import progressbar
import torch
from torch.nn.functional import softmax
from sklearn.decomposition import PCA


class ScReduction(object):
    def __init__(self, args, sc_matrix, labels, bulk_W=None):
        self.X = sc_matrix
        self.labels = labels
        self.M, self.N = self.X.shape
        self.K = args.K
        self.W = np.random.random(size=(self.M, self.K))
        self.H = np.random.random(size=(self.K, self.N))
        self.Z = np.eye(self.N)
        self.bulk_W = bulk_W
        self.e = np.array([1] * self.K).reshape(-1, 1)
        self.gamma = args.gamma
        self.lamb = args.lamb
        self.delta = args.delta
        self.upd_time = args.upd_time
        self.args = args

    def prepare_input(self):
        self.X = torch.from_numpy(self.X).float().cuda()
        self.W = torch.from_numpy(self.W).float().cuda()
        self.H = torch.from_numpy(self.H).float().cuda()
        self.e = torch.from_numpy(self.e).float().cuda()
        if self.Z is not None:
            self.Z = torch.from_numpy(self.Z).float().cuda()
        if self.bulk_W is not None:
            self.bulk_W = torch.from_numpy(self.bulk_W).float().cuda()

    def release(self):
        self.X = self.X.cpu().numpy()
        self.W = self.W.cpu().numpy()
        self.H = self.H.cpu().numpy()
        self.e = self.e.cpu().numpy()
        if self.Z is not None:
            self.Z = self.Z.cpu().numpy()
        if self.bulk_W is not None:
            self.bulk_W = self.bulk_W.cpu().numpy()

    def update(self):
        if self.args.method == "sc-only":
            self.W = self.W * (self.X.mm(self.H.T) / self.W.mm(self.H).mm(self.H.T))
            self.H = self.H * (self.W.T.mm(self.X) / (
                    self.W.T.mm(self.W).mm(self.H) + self.gamma * self.e.mm(self.e.T).mm(self.H)))
        elif self.args.method == "bulk-only":
            self.H = self.H * (self.bulk_W.T.mm(self.X) / (
                    self.bulk_W.T.mm(self.bulk_W).mm(self.H) + self.gamma * self.e.mm(self.e.T).mm(self.H)))
        elif self.args.method == "bulk-sc":
            self.W = self.W * ((self.lamb * self.bulk_W + self.X.mm(self.H.T)) / (
                    self.lamb * self.W + self.W.mm(self.H).mm(self.H.T)))
            self.H = self.H * (self.W.T.mm(self.X) / (
                    self.W.T.mm(self.W).mm(self.H) + self.gamma * self.e.mm(self.e.T).mm(self.H)))
        elif self.args.method == "bulk-sc-aug":
            self.W = self.W * ((self.lamb * self.bulk_W + self.X.mm(self.H.T)) / (
                    self.lamb * self.W + self.W.mm(self.H).mm(self.H.T)))
            self.H = self.H * ((self.W.T.mm(self.X).mm(self.Z) + self.delta * self.H.mm(self.Z + self.Z.T)) / (
                (self.W.T.mm(self.W) + 2 * self.delta * self.H.mm(self.H.T) + self.gamma * self.e.mm(self.e.T)).mm(
                    self.H)))
            for _ in range(100):
                self.Z = self.Z * ((self.X.T.mm(self.W).mm(self.H) + self.delta * self.H.T.mm(self.H)) / (
                        self.X.T.mm(self.X).mm(self.Z) + self.delta * self.Z))
            # normalize
            self.H = softmax(self.H, dim=1)
            self.Z = softmax(self.Z, dim=0)
        elif self.args.method == "pca":
            pca = PCA(n_components=self.K)
            self.H = pca.fit_transform(self.X.T).T
        else:
            raise KeyError("No method called: ", self.args.method)

    def train(self):
        p = progressbar.ProgressBar()
        if self.args.method != "pca":
            # self.prepare_input()
            print("sc model start training...")
            for _ in p(range(self.upd_time)):
                self.update()
            # self.release()
        else:
            print("sc model start training...")
            self.update()
        print("sc model train over.")

    def get_w(self):
        return self.W

    def get_h(self):
        return self.H

    def get_Z(self):
        return self.Z

    def get_data(self):
        return self.X, self.labels

    def set_bulk_W(self, bulk_W):
        self.bulk_W = bulk_W
