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

        # set up the train operation
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)

        # the saver for handling all parameters for the model
        self.saver = tf.train.Saver(tf.global_variables())

    def step(self, session, inputs):
        feed_dict = {}
        for i in range(len(inputs)):
            feed_dict[self.encoder_inputs[i].name] = inputs[i]

        loss, _ = session.run([self.loss, self.train_op], feed_dict=feed_dict)
        return loss

    def get_batch(self, data):
        """ Get a random batch of data, prepare for step.
        :param data: 
        :return: the encoder_inputs for the model
        """
        pass


if __name__ == '__main__':
    model = Seq2SeqModel(10, 1000, 1024, num_layers=2, batch_size=32, learning_rate=0.1)
