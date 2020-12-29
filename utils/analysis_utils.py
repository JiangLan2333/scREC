#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/29 上午9:46
# @Author  : Lan Jiang
# @File    : analysis_utils.py

import pickle


def fetch_low_dimension_data(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    h = model.get_h()
    _, labels = model.get_data()
    return h, labels
