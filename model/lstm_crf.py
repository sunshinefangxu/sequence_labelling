# -*- coding: utf-8 -*-
# @Time    : 2018/5/30 下午2:54
# @Author  : lfx
# @FileName: lstm_crf.py
# @Software: PyCharm

import tensorflow as tf
import copy
import interface.model as model


def model_graph(features, mode, params):

    return ""

class LSTM_CRF(model.Model):

    def __init__(self, params, scope='lstm_crf'):
        super(LSTM_CRF, self).__init__(params=params, scope=scope)

    def get_training_func(self, initializer, regularizer=None):
        def training_fn(features, params=None, reuse=None):
            if params is None:
                params = self.parameters
            with tf.variable_scope(self._scope, initializer=initializer,
                                   regularizer=regularizer, reuse=reuse):
                loss = model_graph(features, "train", params)
                return loss

        return training_fn

    def get_evaluation_func(self):
        def evaluation_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            params.dropout = 0.0
            params.use_variational_dropout = False
            params.label_smoothing = 0.0

            with tf.variable_scope(self._scope):
                score = model_graph(features, "eval", params)

            return score

        return evaluation_fn

    def get_inference_func(self):
        def inference_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            params.dropout = 0.0
            params.use_variational_dropout = False
            params.label_smoothing = 0.0

            with tf.variable_scope(self._scope):
                log_prob = model_graph(features, "infer", params)

            return log_prob

        return inference_fn

    @staticmethod
    def get_name():
        return "lstm_crf"

    @staticmethod
    def get_parameters():
        params = tf.contrib.training.HParams(
            # vocabulary
            pad="<pad>",
            unk="<unk>",
            eos="<eos>",
            bos="<eos>",
            append_eos=False,
            # model
            rnn_cell="LSTM",
            embedding_size=100,
            hidden_size=100,
            # regularization
            dropout=0.2,
            constant_batch_size=True,
            batch_size=128,
            max_length=60,
            clip_grad_norm=5.0
        )

        return params