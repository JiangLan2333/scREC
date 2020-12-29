#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/28 上午10:38
# @Author  : Lan Jiang
# @File    : run_reduction.py

import argparse

from sklearn.manifold import TSNE

from models.bulkReduction import BulkReduction
from models.scReduction import ScReduction
from utils.data_utils import *
from utils.visualization import *


METHODS = ["pca", "sc-only", "bulk-only", "bulk-sc", "bulk-sc-aug"]


def main():
    parser = argparse.ArgumentParser()
    # method & task
    parser.add_argument('--task_name', default="insilico", type=str,
                        help="choose dataset from: {}".format(TASKS2FILES.keys()))
    parser.add_argument('--method', default="bulk-sc", type=str,
                        help="choose method from: {}".format(",".join(METHODS)))

    # hyper parameters
    parser.add_argument('--K', default=20, type=int)
    parser.add_argument('--upd_time', default=1000, type=int)
    parser.add_argument('--gamma', default=1.0, type=float)
    parser.add_argument('--delta', default=100.0, type=float)
    parser.add_argument('--lamb', default=1.0, type=float)

    # load & save
    parser.add_argument('--data_dir', default="/home/chenshengquan/data/scCASimp_data", type=str)
    parser.add_argument('--result_dir', default="./result", type=str)
    parser.add_argument('--aux_dir', default="./aux_files", type=str)

    args = parser.parse_args()

    print("="*10, "TASK: ", args.task_name, " METHOD: ", args.method, "="*10)
    # load data & model
    # sc data & model
    sc_save_dir = os.path.join(args.result_dir, args.task_name, args.method)
    suffix = get_sc_model_suffix(args)
    sc_aux_file = os.path.join(sc_save_dir, "sc_model_{}.pickle".format(suffix))
    if os.path.exists(sc_aux_file):
        print("load pre-trained sc model from {}".format(sc_aux_file))
        with open(sc_aux_file, "rb") as f:
            sc_model = pickle.load(f)
        sc_matrix, labels = sc_model.get_data()
    else:
        print("Initialize sc model.")
        sc_matrix, labels = load_sc_data(args)
        sc_model = ScReduction(args, sc_matrix, labels)

        # bulk data and model
        bulk_W = None
        if args.method not in ["sc-only", "pca"]:
            save_dir = os.path.join(args.result_dir, args.task_name)
            bulk_aux_file = os.path.join(save_dir, "bulk_model_{}.pickle".format(args.gamma))
            if os.path.exists(bulk_aux_file):
                print("load pre-trained bulk model from {}".format(bulk_aux_file))
                with open(bulk_aux_file, "rb") as f:
                    bulk_model = pickle.load(f)
                bulk_W = bulk_model.get_w()
            else:
                print("Initialize bulk model.")
                bulk_matrix = load_bulk_data(args)
                bulk_model = BulkReduction(args, bulk_matrix)

        # train
        if args.method in ["pca", "sc-only"]:
            sc_model.train()
        else:
            if bulk_W is None:
                bulk_model.train()
                save_model(bulk_model, args, "bulk")
                bulk_W = bulk_model.get_w()

            sc_model.set_bulk_W(bulk_W)
            sc_model.train()
        save_model(sc_model, args, "sc")

    # dimension reduction
    sc_H = sc_model.get_h()
    if os.path.exists(os.path.join(sc_save_dir, 'sc_tsne.out')):
        print("load TSNE result from: ", os.path.join(sc_save_dir, 'sc_tsne.out'))
        sc_tsne = np.loadtxt(os.path.join(sc_save_dir, 'sc_tsne.out'), delimiter=",", dtype=float)
    else:
        print("reduce dimension for visualization...")
        tsne_trans = TSNE(n_components=2)
        try:
            sc_tsne = tsne_trans.fit_transform(sc_H.T)
            print("reduce over.")
        except ValueError:
            print("*"*9, "ERROR", "*"*9)
            print("method {} is invalid for task {}".format(args.method, args.task_name))
            return

    # save H matrix and labels for analysis
    np.savetxt(os.path.join(sc_save_dir, 'sc_H.out'), sc_H, delimiter=',')
    np.savetxt(os.path.join(sc_save_dir, 'sc_tsne.out'), sc_tsne, delimiter=',')
    np.savetxt(os.path.join(sc_save_dir, 'sc_labels.out'), labels, delimiter=',', fmt="%s")
    print("save H, TSNE, labels for single cells over.")
    # visualization
    pic_path = os.path.join(args.result_dir, args.task_name, args.method, "sc_{}.png".format(suffix))
    draw_and_save_figure(sc_tsne, labels, pic_path)


if __name__ == "__main__":
    main()
