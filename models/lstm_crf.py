# -*- coding: utf-8 -*-
# @Time    : 2018/5/30 下午2:54
# @Author  : lfx
# @FileName: lstm_crf.py
# @Software: PyCharm

import tensorflow as tf
import copy
import interface.model as model

def encoding_graph(features, mode, params):

    if mode != 'train':
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0
    hidden_size = params.hidden_size
    src_seq = features["source"]
    src_len = features["source_length"]
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=tf.float32)

    svocab = params.vocabulary["source"]
    src_vocab_size = len(svocab)

    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    src_embedding = tf.get_variable("weights", [src_vocab_size, hidden_size], initializer=initializer)
    bias = tf.get_variable("bias", [hidden_size])

    inputs = tf.gather(src_embedding, src_seq)


def _lstm_encoder(cell, inputs, sequence_length, initial_state, dtype=None):

    a = 1

def _encoder(cell_fw, cell_bw, inputs, sequence_length, dtype=None,
             scope=None):
    with tf.variable_scope(scope or "encoder",
                           values=[inputs, sequence_length]):
        inputs_fw = inputs

        inputs_bw = tf.reverse_sequence(inputs, sequence_length, batch_axis=0, seq_axis=1)

        with tf.variable_scope("forward"):
            output_fw, state_fw = _lstm_encoder(cell_fw, inputs_fw, sequence_length, None, dtype=dtype)

        with tf.variable_scope("backward"):
            output_bw, state_bw = _lstm_encoder(cell_bw, inputs_bw,
                                                sequence_length, None,
                                                dtype=dtype)

            output_bw = tf.reverse_sequence(output_bw, sequence_length,
                                            batch_axis=0, seq_axis=1)



def model_graph(features, mode, params):

    src_vocab_size = len(params.vocabulary["source"])
    tgt_vocab_size = len(params.vocabulary["target"])

    with tf.variable_scope("source_embedding"):
        src_emb = tf.get_variable("embedding",
                                  [src_vocab_size, params.embedding_size])

        src_bias = tf.get_variable("bias", [params.embedding_size])
        src_inputs = tf.nn.embedding_lookup(src_emb, features["source"])

        src_inputs = tf.nn.bias_add(src_inputs, src_bias)
        if params.dropout and not params.use_variational_dropout:
            src_inputs = tf.nn.dropout(src_inputs, 1-params.dropout)

        #encoder

        cell_fw = tf.contrib.rnn.LSTMCell(params.hidden_size, forget_bias=0.0)
        cell_bw = tf.contrib.rnn.LSTMCell(params.hidden_size, forget_bias=0.0)

        if params.use_variational_dropout:
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(
                cell_fw,
                input_keep_prob=1 - params.dropout,
                output_keep_prob=1 - params.dropout,
                state_keep_prob=1 - params.dropout,
                variational_recurrent=True,
                input_size=params.embedding_size,
                dtype=tf.float32
            )

            cell_bw = tf.nn.rnn_cell.DropoutWrapper(
                cell_bw,
                input_keep_prob=1 - params.dropout,
                output_keep_prob=1 - params.dropout,
                state_keep_prob=1 - params.dropout,
                variational_recurrent=True,
                input_size=params.embedding_size,
                dtype=tf.float32
            )

        encoder_output = _encoder(cell_fw, cell_bw, src_inputs, features["source_length"])
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