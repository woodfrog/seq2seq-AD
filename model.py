import random

import numpy as np
import tensorflow as tf

from tensorflow.contrib.legacy_seq2seq import basic_rnn_seq2seq
from tensorflow.contrib.rnn import BasicLSTMCell, MultiRNNCell


class Seq2SeqModel:
    def __init__(self,
                 data_size,
                 time_len,
                 unit_size,
                 num_layers,
                 batch_size,
                 learning_rate):
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

        # Set placeholder for encoder's inputs
        self.encoder_inputs = []
        self.decoder_inputs = []

        for i in range(self.time_len):
            self.encoder_inputs.append(tf.placeholder(
                shape=[self.batch_size, self.input_size], name='encoder{}'.format(i),
                dtype=tf.float32))

            self.decoder_inputs.append(tf.zeros(shape=[self.batch_size, self.input_size]))

        # The purpose is reconstruction, thus the targets should be the reverse of the input
        targets = self.encoder_inputs[::-1]

        hidden_states, _ = basic_rnn_seq2seq(self.encoder_inputs, self.decoder_inputs, cell)

        w_out = tf.get_variable(name='w_out', shape=[unit_size, self.input_size])
        b_out = tf.get_variable(name='b_out', shape=[self.input_size])

        outputs = []
        for i in range(time_len):
            outputs.append(tf.matmul(hidden_states[i], w_out) + b_out)

        targets = tf.stack(targets, axis=1)
        self.outputs = tf.stack(outputs, axis=1)
        self.loss = tf.losses.mean_squared_error(targets, self.outputs)
        self.error_vector = tf.abs(self.outputs - targets)

        # set up the train operation
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

        # the saver for handling all parameters for the model
        self.saver = tf.train.Saver(tf.global_variables())

    def step(self, session, inputs, train=True):
        """
        run one step of training using the given session and inputs
        :param session: the session to run the step
        :param inputs: a list, each element has shape ()
        :return: 
        """
        feed_dict = {}

        if len(inputs) != self.time_len:
            raise ValueError('The length of inputs is {}, but it must be the same as '
                             'model\'s time length, which is {}'.format(len(inputs), self.time_len))

        for i in range(len(inputs)):
            feed_dict[self.encoder_inputs[i].name] = inputs[i]

        if train:
            loss, _ = session.run([self.loss, self.train_op], feed_dict=feed_dict)
        else:  # during validation, not running the training operation
            loss = session.run(self.loss, feed_dict=feed_dict)

        return loss

    def get_err_vec(self, session, inputs):
        feed_dict = {}

        if len(inputs) != self.time_len:
            raise ValueError('The length of inputs is {}, but it must be the same as '
                             'model\'s time length, which is {}'.format(len(inputs), self.time_len))

        for i in range(len(inputs)):
            feed_dict[self.encoder_inputs[i].name] = inputs[i]

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
                             'from the model\'s specified input size {}'.format(data[0].shape[0], self.input_size))

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
        for t in range(self.time_len):
            data_t = []
            for idx in indices:
                data_t.append(data[idx][t])
            # now data_t is of length batch_size
            data_t = np.stack(data_t, axis=0)
            encoder_inputs.append(data_t)
        # print(len(encoder_inputs))
        # print(encoder_inputs[0].shape)
        return encoder_inputs

        # if __name__ == '__main__':
        # model = Seq2SeqModel(10, 5, 100, num_layers=2, batch_size=32, learning_rate=0.1)
        # model.get_batch([np.random.randn(5) for _ in range(1000)])
