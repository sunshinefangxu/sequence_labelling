# -*- coding: utf-8 -*-
# @Time    : 2018/5/30 上午11:36
# @Author  : lfx
# @FileName: __init__.py.py
# @Software: PyCharm

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import model.lstm_crf
import model.bi_lstm_crf
import model.lstm_cnn

def get_model(name):

    name = name.lower()

    if name == "lstm_crf":
        return model.lstm_crf
    elif name == "bi_lstm_crf":
        return model.bi_lstm_crf
    elif name == "lstm_cnn":
        return model.lstm_cnn
    else:
        raise LookupError("Unknown model %s" % name)
