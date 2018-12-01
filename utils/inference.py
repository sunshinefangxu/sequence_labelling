# -*- coding: utf-8 -*-
# @Time    : 2018/6/9 下午1:58
# @Author  : lfx
# @FileName: inference.py
# @Software: PyCharm

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf

from collections import namedtuple
from tensorflow.python.util import nest


def create_inference_graph(models, features, params):

    if not isinstance(models, (list, tuple)):
        raise ValueError("'models' must be a list or tuple")

    features = copy.copy(features)
    model_fns = [model.get_inference_func() for model in models]

    # Compute initial state if necessary
    states = []
    funcs = []

    for model_fn in model_fns:
        if callable(model_fn):
            # For non-incremental decoding
            states.append({})
            funcs.append(model_fn)
        else:
            # For incremental decoding where model_fn is a tuple:
            # (encoding_fn, decoding_fn)
            states.append(model_fn[0](features))
            funcs.append(model_fn[1])

    batch_size = tf.shape(features["source"])[0]
    # 0 1 1

    model_fn = funcs[0]
    unary_scores, transition_params, sequence_len = model_fn(features)

    viterbi_sequence, _ = tf.contrib.crf.crf_decode(
        unary_scores, transition_params, sequence_len)

    tgt_mask = tf.sequence_mask(sequence_len, maxlen=tf.shape(unary_scores)[1])
    tgt_mask = tf.cast(tgt_mask, tf.int32)
    viterbi_sequence = tf.multiply(viterbi_sequence, tgt_mask)
    viterbi_sequence = tf.expand_dims(viterbi_sequence, 1)
    return viterbi_sequence

if __name__=='__main__':

    with tf.Session() as sess:
        features = {}
        temp = tf.random_uniform([2,5,4], minval=-0.6,maxval=0.6)
        features["target"] = tf.constant([[1,2,0,2,3],[0,3,2,1, 1]], dtype=tf.int32)
        features["source_length"] = tf.constant([5,4], dtype=tf.int32)
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
            temp, features["target"], features["source_length"])

        score = tf.reduce_mean(log_likelihood)
        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(score)
        init = tf.global_variables_initializer()
        sess.run(init)
        temp = sess.run(temp)
        a = tf.convert_to_tensor(temp)
        mask = tf.sequence_mask(features["source_length"], maxlen=tf.shape(a)[1])
        print(sess.run(mask))
        mask = tf.cast(mask, tf.int32)
        viterbi_sequence, _ = tf.contrib.crf.crf_decode(a, transition_params, features["source_length"])
        seq, _ = sess.run([viterbi_sequence, train_op])
        seq = tf.multiply(seq, mask)
        print(sess.run(seq))
        features["source_length"] = sess.run(features["source_length"])
        # print(len(temp))
        print(features["source_length"])
        transition_params = sess.run(transition_params)
        for un, length in zip(temp, features["source_length"]):
            un = un[:length]
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                un, transition_params)

            print(viterbi_sequence)
