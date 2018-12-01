# -*- coding: utf-8 -*-
# @Time    : 2018/5/30 下午2:54
# @Author  : lfx
# @FileName: lstm_crf.py
# @Software: PyCharm

import tensorflow as tf
import copy
import thumt.interface as interface


def _copy_through(time, length, output, new_output):
    copy_cond = (time >= length)
    return tf.where(copy_cond, output, new_output)


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


def _lstm_encoder(cell, inputs, sequence_length, initial_state, dtype):
    output_size = cell.output_size
    dtype = dtype or inputs.dtype
    batch = tf.shape(inputs)[0]
    time_steps = tf.shape(inputs)[1]

    zero_output = tf.zeros([batch, output_size], dtype)
    if initial_state is None:
        initial_state = cell.zero_state(batch, dtype)

    input_ta = tf.TensorArray(dtype, time_steps,
                              tensor_array_name="input_array")
    output_ta = tf.TensorArray(dtype, time_steps,
                               tensor_array_name="output_array")
    input_ta = input_ta.unstack(tf.transpose(inputs, [1, 0, 2]))

    def loop_func(t, out_ta, state):
        inp_t = input_ta.read(t)
        cell_output, new_state = cell(inp_t, state)
        cell_output = _copy_through(t, sequence_length, zero_output,
                                    cell_output)
        # new_state = _copy_through(t, sequence_length, state, new_state)
        out_ta = out_ta.write(t, cell_output)
        return t + 1, out_ta, new_state

    time = tf.constant(0, dtype=tf.int32, name="time")
    loop_vars = (time, output_ta, initial_state)

    outputs = tf.while_loop(lambda t, *_: t < time_steps, loop_func,
                            loop_vars, parallel_iterations=32,
                            swap_memory=True)

    output_final_ta = outputs[1]
    final_state = outputs[2]

    all_output = output_final_ta.stack()
    all_output.set_shape([None, None, output_size])
    all_output = tf.transpose(all_output, [1, 0, 2])

    return all_output, final_state


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

        results = {
            # [batch, maxlen, 2*hidden]
            "annotation": tf.concat([output_fw, output_bw], axis=2),
            "outputs": {
                # [batch,maxlen,hidden]
                "forward": output_fw,
                "backward": output_bw
            },
            "final_states": {
                # [batch,hidden]
                "forward": state_fw,
                "backward": state_bw
            }
        }

        return results


def _char_convolution(inputs, cfilter):
    conv1 = tf.nn.conv2d(inputs, cfilter, [1, 1, 1, 1],
                         padding='VALID')
    # inputs.shape = [batch_size, 1, 3, 2*num_units]
    # namely, in_height = 1, in_width = 3, in_channels = 2*num_units
    # filter.shape = [1, 2, 2*num_units, num_tags], strides = [1, 1, 1, 1]
    # conv1.shape = [batch_size, 1, 2, num_tags]
    conv1 = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool(conv1,
                           ksize=[1, 1, 2, 1],
                           strides=[1, 1, 1, 1],
                           padding='VALID')

    # pool1.shape = [batch_size, 1, 1, num_tags]
    pool1 = tf.squeeze(pool1, [1, 2])
    # pool1.shape = [batch_size, num_tags]
    return pool1


def create_cnn_layer(inputs, time_major, batch_size, num_units, cfilter):
    if not time_major:
        # trnaspose
        inputs = tf.transpose(inputs, [1, 0, 2])

    inputs = tf.expand_dims(inputs, 2)
    # [max_time, batch_size, 1, 2*num_units]

    left = inputs[1:]
    right = inputs[: -1]
    left = tf.pad(left, [[1, 0], [0, 0], [0, 0], [0, 0]], "CONSTANT")
    right = tf.pad(right, [[0, 1], [0, 0], [0, 0], [0, 0]], "CONSTANT")

    char_blocks = tf.concat([left, inputs, right], 3)
    # [max_time, batch_size, 1, 3*2*num_units]
    char_blocks = tf.reshape(char_blocks, [-1, batch_size, 3, (2 * num_units)])
    char_blocks = tf.expand_dims(char_blocks, 2)
    # [max_time, batch_size, 1, 3, 2*num_units]

    do_char_conv = lambda x: _char_convolution(x, cfilter)
    cnn_outputs = tf.map_fn(do_char_conv, char_blocks)
    # [max_time, batch_size, num_tags]

    return tf.transpose(cnn_outputs, [1, 0, 2])


def _cnn_layer(encoder_outputs, params, dtype=None):
    batch_size = tf.shape(encoder_outputs)[0]
    num_units = params.hidden_size
    num_tags = len(params.vocabulary["target"])
    # CNN
    with tf.variable_scope('cnn', dtype=dtype) as scope:
        cfilter = tf.get_variable(
            "cfilter",
            shape=[1, 2, 2 * num_units, num_tags],
            regularizer=tf.contrib.layers.l2_regularizer(0.0001),
            initializer=tf.truncated_normal_initializer(stddev=params.filter_init_std),
            dtype=tf.float32)

    return create_cnn_layer(encoder_outputs, False,
                            batch_size, num_units, cfilter)


def model_graph(features, mode, params):
    src_vocab_size = len(params.vocabulary["source"])
    tgt_vocab_size = len(params.vocabulary["target"])

    with tf.variable_scope("source_embedding"):
        src_emb = tf.get_variable("embedding",
                                  [src_vocab_size, params.embedding_size])

        src_bias = tf.get_variable("bias", [params.embedding_size])
        src_inputs = tf.nn.embedding_lookup(src_emb, features["source"])
        time_steps = tf.shape(src_inputs)[1]
        src_inputs = tf.nn.bias_add(src_inputs, src_bias)
        if params.dropout and not params.use_variational_dropout:
            src_inputs = tf.nn.dropout(src_inputs, 1 - params.dropout)

        # encoder

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

        output = encoder_output["annotation"]
        output = _cnn_layer(output, params)
        # output = tf.reshape(output, [-1, params.hidden_size * 2])

        # if mode == 'train':
        #    output = tf.nn.dropout(output, params.dropout)
        '''
        with tf.variable_scope("softmax"):
            weights = tf.get_variable("weights", [params.hidden_size * 2, tgt_vocab_size])
            bias = tf.get_variable("bias", [tgt_vocab_size])

        matricized_unary_scores = tf.matmul(output, weights) + bias
        '''
        matricized_unary_scores = output
        unary_scores = tf.reshape(
            matricized_unary_scores,
            [-1, time_steps, tgt_vocab_size])

        if mode is "infer":
            features["target"] = features["source"]
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
                unary_scores, features["target"], features["source_length"])

            return unary_scores, transition_params, features["source_length"]

        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
            unary_scores, features["target"], features["source_length"])

        total_loss = tf.reduce_mean(-log_likelihood)

    return total_loss


class LSTM_CNN(interface.NMTModel):

    def __init__(self, params, scope='lstm_crf'):
        super(LSTM_CNN, self).__init__(params=params, scope=scope)

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
                unary_scores, transition_params, sequence_len = model_graph(features, "infer", params)

            return unary_scores, transition_params, sequence_len

        return inference_fn

    @staticmethod
    def get_name():

        return "lstm_cnn"

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
            filter_init_std=0.035,
            embedding_size=100,
            hidden_size=100,
            use_variational_dropout=False,
            # regularization
            dropout=0.2,
            constant_batch_size=True,
            batch_size=128,
            max_length=100,
            clip_grad_norm=5.0
        )

        return params
