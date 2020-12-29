#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/28 上午10:42
# @Author  : Lan Jiang
# @File    : data_utils.py

import os
import pickle

import hdf5storage
import numpy as np

TASKS2FILES = {"tissue": ["tissue4_forebrain.mat", "MCA_bulk_mat.mat"],
               "forebrain": ["forebrain_half.mat", "forebrain_bulk_mat.mat"],
               "gmvshek": ["GMvsHek.mat", "GMvsHek_AllRO_bulk_mat.mat"],
               "gmvshl": ["GMvsHL.mat", "GMvsHL_AllRO_bulk_mat.mat"],
               "insilico": ["InSilico.mat", "InSilico_AllRO_bulk_mat.mat"]}


def load_sc_data(args):
    aux_dir = os.path.join(args.aux_dir, args.task_name)
    aux_matrix_file = os.path.join(aux_dir, "sc_matrix.pickle")
    aux_label_file = os.path.join(aux_dir, "sc_label.pickle")
    if os.path.exists(aux_matrix_file) and os.path.exists(aux_label_file):
        print("load sc data from aux file {} and {}".format(aux_matrix_file, aux_label_file))
        with open(aux_matrix_file, "rb") as f:
            matrix = pickle.load(f)
        with open(aux_label_file, "rb") as f:
            labels = pickle.load(f)
    else:
        file_name = TASKS2FILES[args.task_name][0]
        print("load sc data from origin file {}".format(file_name))
        data = hdf5storage.loadmat(os.path.join(args.data_dir, file_name))
        matrix = data['count_mat']
        # fetch label with regards to different dataset.
        labels = np.squeeze(data['label_mat'])
        if args.task_name == "insilico":
            labels = [labels[i][0] for i in range(len(labels))]
        if not os.path.exists(aux_dir):
            os.mkdir(aux_dir)
        pickle.dump(matrix, open(aux_matrix_file, "wb"), protocol=4)
        pickle.dump(labels, open(aux_label_file, "wb"), protocol=4)

    return matrix, labels


def load_bulk_data(args):
    aux_matrix_file = os.path.join(args.aux_dir, args.task_name, "bulk_matrix.pickle")
    if os.path.exists(aux_matrix_file):
        print("load bulk data from aux file {}".format(aux_matrix_file))
        with open(aux_matrix_file, "rb") as f:
            matrix = pickle.load(f)
    else:
        file_name = TASKS2FILES[args.task_name][1]
        print("load bulk data from origin file {}".format(file_name))
        data = hdf5storage.loadmat(os.path.join(args.data_dir, file_name))
        matrix = data['bulk_mat'].T
        with open(aux_matrix_file, "wb") as f:
            pickle.dump(matrix, f)

    return matrix


def get_sc_model_suffix(args):
    if args.method in ["sc-only", "bulk-only"]:
        suffix = str(args.gamma)
    elif args.method == "bulk-sc":
        suffix = '_'.join([str(args.gamma), str(args.lamb)])
    elif args.method == "bulk-sc-aug":
        suffix = '_'.join([str(args.gamma), str(args.lamb), str(args.delta)])
    elif args.method == "pca":
        suffix = "pca"
    else:
        suffix = None
    return suffix


def save_model(model, args, model_type):
    if model_type == "bulk":
        save_dir = os.path.join(args.result_dir, args.task_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        with open(os.path.join(save_dir, "bulk_model_{}.pickle".format(args.gamma)), "wb") as f:
            pickle.dump(model, f)
        print("save bulk model to {}.".format(os.path.join(save_dir, "bulk_model_{}.pickle".format(args.gamma))))
    elif model_type == "sc":
        save_dir = os.path.join(args.result_dir, args.task_name, args.method)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        suffix = get_sc_model_suffix(args)
        with open(os.path.join(save_dir, "sc_model_{}.pickle".format(suffix)), "wb") as f:
            pickle.dump(model, f)
        print("save sc model to {}.".format(os.path.join(save_dir, "sc_model_{}.pickle".format(suffix))))
