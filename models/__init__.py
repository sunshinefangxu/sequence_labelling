# -*- coding: utf-8 -*-
# @Time    : 2018/5/30 上午11:36
# @Author  : lfx
# @FileName: __init__.py.py
# @Software: PyCharm

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import models.lstm_crf
import models.bi_lstm_crf
import models.lstm_cnn

def get_model(name):

    name = name.lower()

    if name == "lstm_crf":
        return models.lstm_crf.LSTM_CRF
    elif name == "bi_lstm_crf":
        return models.bi_lstm_crf
    elif name == "lstm_cnn":
        return models.lstm_cnn
    else:
        raise LookupError("Unknown model %s" % name)
