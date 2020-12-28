#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/28 上午10:10
# @Author  : Lan Jiang
# @File    : visualization.py

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


def draw_and_save_figure(low_dim_data, labels, save_path):
    fig = plt.figure()
    fig.set_figwidth(8)
    fig.set_figheight(8)
    df = {'component_1': low_dim_data[:, 0], 'component_2': low_dim_data[:, 1], 'label': labels}
    df = pd.DataFrame(df)
    ax = sns.scatterplot(x="component_1", y="component_2", hue="label", s=50, data=df)
    ax.legend()
    plt.savefig(save_path)
    print("save pic to {}".format(save_path))