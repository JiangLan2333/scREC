#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/28 上午10:40
# @Author  : Lan Jiang
# @File    : bulkReduction.py

import numpy as np
from numpy.linalg import multi_dot
import progressbar
import os
import pickle


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

    def update(self):
        self.W = self.W * (self.X.dot(self.H.T) / multi_dot([self.W, self.H, self.H.T]))
        self.H = self.H * (self.W.T.dot(self.X) / (
                    multi_dot([self.W.T, self.W, self.H]) + self.gamma * multi_dot([self.e, self.e.T, self.H])))

    def train(self):
        print("bulk model start training...")
        p = progressbar.ProgressBar()
        for _ in p(range(self.upd_time)):
            self.update()
        print("bulk model train over.")

    def get_w(self):
        return self.W

    def get_h(self):
        return self.H