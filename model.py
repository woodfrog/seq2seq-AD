import random

import numpy as np
import tensorflow as tf
import copy

from tensorflow.contrib.legacy_seq2seq import rnn_decoder
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.rnn import BasicLSTMCell, MultiRNNCell
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn

'''

https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/legacy_seq2seq/python/ops/seq2seq.py

'''


def _extract_last_and_project(output_projection=None):
    def loop_function(prev, _):
        if output_projection is not None:
            prev = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])
        return prev

    return loop_function


def advanced_rnn_decoder(decoder_inputs,
                         initial_state,
                         cell,
                         num_symbols,
                         output_projection=None,
                         feed_previous=False,
                         scope=None):
    with variable_scope.variable_scope(scope or "embedding_rnn_decoder") as scope:
        if output_projection is not None:
            dtype = scope.dtype
            proj_weights = ops.convert_to_tensor(output_projection[0], dtype=dtype)
            proj_weights.get_shape().assert_is_compatible_with([None, num_symbols])
            proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
            proj_biases.get_shape().assert_is_compatible_with([num_symbols])

        if feed_previous:
            loop_function = _extract_last_and_project(output_projection)
        else:
            loop_function = None

        return rnn_decoder(
            decoder_inputs, initial_state, cell, loop_function=loop_function)


def advanced_rnn_seq2seq(encoder_inputs,
                         decoder_inputs,
                         cell,
                         num_decoder_symbols,
                         output_projection=None,
                         feed_previous=False,
                         dtype=None,
                         scope=None):
    with variable_scope.variable_scope(scope or "advanced_rnn_seq2seq") as scope:
        if dtype is not None:
            scope.set_dtype(dtype)
        else:
            dtype = scope.dtype

        # Encoder.
        encoder_cell = copy.deepcopy(cell)  # different weights, so use deepcopy here

        _, encoder_state = rnn.static_rnn(encoder_cell, encoder_inputs, dtype=dtype)

        print('encoder state', encoder_state[-1].h, type(encoder_state[-1]), sep='\n')
        # print(encoder_state)

        # Decoder.

        # if projection weights are not provided, automatically add with wrapper
        if output_projection is None:
            cell = core_rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)

        return advanced_rnn_decoder(
            decoder_inputs,
            encoder_state,
            cell,
            num_decoder_symbols,
            output_projection=output_projection,
            feed_previous=feed_previous)


class Seq2SeqModel:
    def __init__(self,
                 data_size,
                 time_len,
                 unit_size,
                 num_layers,
                 batch_size,
                 learning_rate,
                 feed_previous):
        '''
        Create the basic encoder-decoder seq2seq model
        :param unit_size: number of units in each LSTM layer of the model
        :param num_layers: number of LSTM layers in the model
        :param batch_size: the size of batches used during training
        :param learning_rate: 
        '''

        self.input_size = data_size
        self.time_len = time_len
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(
            float(learning_rate), trainable=False, name='lr')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        def single_cell():
            return BasicLSTMCell(unit_size)

        cell = single_cell()
        if num_layers > 1:
            cell = MultiRNNCell([single_cell() for _ in range(num_layers)])

        print('state size', cell.state_size)
        print('zero state size', cell.zero_state(self.batch_size, dtype=tf.float32))

        # Set placeholder for encoder's inputs
        self.encoder_inputs = []
        self.decoder_inputs = []

        for i in range(self.time_len):
            self.encoder_inputs.append(tf.placeholder(
                shape=[self.batch_size, self.input_size], name='encoder{}'.format(i),
                dtype=tf.float32))

            self.decoder_inputs.append(tf.placeholder(
                shape=[self.batch_size, self.input_size], name='decoder{}'.format(i),
                dtype=tf.float32
            ))

        # The purpose is reconstruction, thus the targets should be the reverse of the input
        targets = self.encoder_inputs[::-1]
        outputs, _ = advanced_rnn_seq2seq(
            encoder_inputs=self.encoder_inputs,
            decoder_inputs=self.decoder_inputs,
            cell=cell,
            num_decoder_symbols=self.input_size,
            output_projection=None,
            feed_previous=feed_previous
        )  # the outputs have been projected based on the original lstm outputs

        targets = tf.stack(targets, axis=1)
        self.outputs = tf.stack(outputs, axis=1)
        self.loss = tf.losses.mean_squared_error(targets, self.outputs)
        self.error_vector = tf.abs(self.outputs - targets)

        # set up the train operation
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

        # the saver for handling all parameters for the model
        self.saver = tf.train.Saver(tf.global_variables())

    def step(self, session, encoder_inputs, decoder_inputs, train=True):
        """
        run one step of training using the given session and inputs
        :param session: the session to run the step
        :param inputs: a list, each element has shape ()
        :return: 
        """
        feed_dict = {}

        if len(encoder_inputs) != self.time_len:
            raise ValueError('The length of inputs is {}, but it must be the same as '
                             'model\'s time length, which is {}'.format(len(encoder_inputs), self.time_len))

        for i in range(len(encoder_inputs)):
            feed_dict[self.encoder_inputs[i].name] = encoder_inputs[i]

        for i in range(len(decoder_inputs)):
            feed_dict[self.decoder_inputs[i].name] = decoder_inputs[i]

        if train:
            loss, _ = session.run([self.loss, self.train_op], feed_dict=feed_dict)
        else:  # during validation, not running the training operation
            loss = session.run(self.loss, feed_dict=feed_dict)

        return loss

    def get_err_vec(self, session, encoder_inputs, decoder_inputs):
        feed_dict = {}

        if len(encoder_inputs) != self.time_len:
            raise ValueError('The length of inputs is {}, but it must be the same as '
                             'model\'s time length, which is {}'.format(len(encoder_inputs), self.time_len))

        for i in range(len(encoder_inputs)):
            feed_dict[self.encoder_inputs[i].name] = encoder_inputs[i]

        for i in range(len(decoder_inputs)):
            feed_dict[self.decoder_inputs[i].name] = decoder_inputs[i]

        errs = session.run(self.error_vector, feed_dict=feed_dict)

        return errs

    def get_batch(self, data, shuffle=True, start=None):
        """ Get a random batch from data, prepare for **step**.
        :param data: a list of data, each of them is of shape (time_len, data_size)
        :return: the encoder_inputs for the model. A list of length time_len, 
                  each element has shape (batch_size, data_size)
        """
        length = len(data)
        # prepare the start index for each batch

        if data[0].shape[1] != self.input_size:  # data[0]'s shape should be (time_len, data_size)
            raise ValueError('The actual input data size is {}, which is different '
                             'from the model\'s specified input size {}'.format(data[0].shape[1], self.input_size))

        indices = []  # get the start index for each batch
        if shuffle:
            for _ in range(self.batch_size):
                indices.append(random.randint(0, length - 1))
        else:
            if start is None:
                raise ValueError('Should specify the start index for getting the batch')
            for i in range(self.batch_size):
                indices.append(start + i)

        encoder_inputs = []
        decoder_inputs = []

        # construct encoder inputs
        for t in range(self.time_len):
            data_t = []
            for idx in indices:
                data_t.append(data[idx][t])
            # now data_t is of length batch_size
            data_t = np.stack(data_t, axis=0)
            encoder_inputs.append(data_t)

        # construct decoder inputs
        for t in range(self.time_len):
            if t == 0:
                decoder_inputs.append(np.zeros(shape=[self.batch_size, self.input_size]))
            else:
                decoder_inputs.append(encoder_inputs[self.time_len - t])

        # print(len(encoder_inputs))
        # print(encoder_inputs[0].shape)
        return encoder_inputs, decoder_inputs


if __name__ == '__main__':
    pass
    # test the whether the model can be build
    model = Seq2SeqModel(10, 5, 100, num_layers=2, batch_size=32, learning_rate=0.1, feed_previous=True)
    print('model built')
