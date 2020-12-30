#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/28 上午10:40
# @Author  : Lan Jiang
# @File    : bulkReduction.py

import numpy as np
import progressbar
import torch


class BulkReduction(object):
    def __init__(self, args, bulk_matrix):
        self.X = bulk_matrix
        self.M, self.N = self.X.shape
        self.K = args.K
        self.W = np.random.random(size=(self.M, self.K))
        self.H = np.random.random(size=(self.K, self.N))
        self.e = np.array([1] * self.K).reshape(-1, 1)
        self.gamma = args.gamma
        self.upd_time = args.upd_time
        self.args = args

    def prepare_input(self):
        self.X = torch.from_numpy(self.X).float().cuda()
        self.W = torch.from_numpy(self.W).float().cuda()
        self.H = torch.from_numpy(self.H).float().cuda()
        self.e = torch.from_numpy(self.e).float().cuda()

    def release(self):
        self.X = self.X.cpu().numpy()
        self.W = self.W.cpu().numpy()
        self.H = self.H.cpu().numpy()
        self.e = self.e.cpu().numpy()

    def update(self):
        self.W = self.W * (self.X.mm(self.H.T) / self.W.mm(self.H).mm(self.H.T))
        self.H = self.H * (self.W.T.mm(self.X) / (
                   self.W.T.mm(self.W).mm(self.H) + self.gamma * self.e.mm(self.e.T).mm(self.H)))

    def train(self):
        p = progressbar.ProgressBar()
        # self.prepare_input()
        print("bulk model start training...")
        for _ in p(range(self.upd_time)):
            self.update()
        # self.release()
        print("bulk model train over.")

    def get_w(self):
        return self.W

    def get_h(self):
        return self.H