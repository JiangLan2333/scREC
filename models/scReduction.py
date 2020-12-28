#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/28 上午10:40
# @Author  : Lan Jiang
# @File    : scReduction.py

import os

import numpy as np
import progressbar
from numpy.linalg import multi_dot


class ScReduction(object):
    def __init__(self, args, sc_matrix, labels, bulk_W=None):
        self.X = sc_matrix
        self.labels = labels
        self.M, self.N = self.X.shape
        self.K = args.K
        self.W = np.random.random(size=(self.M, self.K))
        self.H = np.random.random(size=(self.K, self.N))
        self.Z = np.random.random(size=(self.N, self.N))
        self.bulk_W = bulk_W
        self.e = np.array([1] * self.K).reshape(-1, 1)
        self.gamma = args.gamma
        self.lamb = args.lamb
        self.delta = args.delta
        self.upd_time = args.upd_time
        self.args = args

    def update(self):
        if self.args.method == "sc-only":
            self.W = self.W * (self.X.dot(self.H.T) / multi_dot([self.W, self.H, self.H.T]))
            self.H = self.H * (self.W.T.dot(self.X) / (
                    multi_dot([self.W.T, self.W, self.H]) + self.gamma * multi_dot([self.e, self.e.T, self.H])))
        elif self.args.method == "bulk-only":
            self.H = self.H * (self.bulk_W.T.dot(self.X) / (
                    multi_dot([self.bulk_W.T, self.bulk_W, self.H]) + self.gamma * multi_dot([self.e, self.e.T, self.H])))
        elif self.args.method == "bulk-sc":
            self.W = self.W * ((self.lamb * self.bulk_W + self.X.dot(self.H.T)) / (
                    self.lamb * self.W + multi_dot([self.W, self.H, self.H.T])))
            self.H = self.H * (self.W.T.dot(self.X) / (
                    multi_dot([self.W.T, self.W, self.H]) + self.gamma * multi_dot([self.e, self.e.T, self.H])))
        elif self.args.method == "bulk-sc-aug":
            self.W = self.W * ((self.lamb * self.bulk_W + self.X.dot(self.H.T)) / (
                    self.lamb * self.W + multi_dot([self.W, self.H, self.H.T])))
            self.H = self.H * ((multi_dot([self.W.T, self.X, self.Z]) + self.delta * self.H.dot(self.Z + self.Z.T)) / (
                (self.W.T.dot(self.W) + 2 * self.delta * self.H.dot(self.H.T) + self.gamma * self.e.dot(self.e.T)).dot(
                    self.H)))
            self.Z = self.Z * ((multi_dot([self.X.T, self.W, self.H]) + self.delta * self.H.T.dot(self.H)) / (
                        multi_dot([self.X.T, self.X, self.Z]) + self.delta * self.Z))
        else:
            raise KeyError("No method called: ", self.args.method)

    def train(self):
        print("sc model start training...")
        p = progressbar.ProgressBar()
        for _ in p(range(self.upd_time)):
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