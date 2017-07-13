import time
import pickle
import os
import tensorflow as tf

from model import Seq2SeqModel
import data_utils

tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_string("train_dir", "./train", "Training directory.")
tf.app.flags.DEFINE_integer('total_steps', 10000,
                            'How many training steps to take')
tf.app.flags.DEFINE_integer('batch_size', 32,
                            'the batch size for training')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
tf.app.flags.DEFINE_integer('time_len', 60, 'the length of time window (in minute)')
tf.app.flags.DEFINE_integer('data_size', 6, 'the dimension of each data point')
tf.app.flags.DEFINE_integer('unit_size', 100, 'the size of each LSTM layer (dimension of hidden states)')
tf.app.flags.DEFINE_integer('num_layers', 2, 'the number of LSTM layers in the model')

FLAGS = tf.app.flags.FLAGS


def read_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    features = data['feature']
    return features


def create_model(session):
    model = Seq2SeqModel(data_size=FLAGS.data_size, time_len=FLAGS.time_len,
                         unit_size=FLAGS.unit_size, num_layers=FLAGS.num_layers, batch_size=FLAGS.batch_size,
                         learning_rate=FLAGS.learning_rate)

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):  # restore the pre-trained model if there is
        print("Reading model parameters from {}".format(ckpt.model_checkpoint_path))
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print('Created model with initialization')
        session.run(tf.global_variables_initializer())
    return model


def train():
    with tf.Session() as sess:
        model = create_model(sess)
        print('model created')

        features = read_data('train_0_0.pickle')
        step_time, loss = 0.0, 0.0
        current_step = 0

        for _ in range(FLAGS.total_steps):
            start_time = time.time()
            encoder_inputs = model.get_batch(features)
            step_loss = model.step(session=sess, inputs=encoder_inputs)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            if current_step % FLAGS.steps_per_checkpoint == 0:
                print('global step {}, step time {}, loss {}'.format(model.global_step.eval(),
                                                                     step_time, loss))
                checkpoint_path = os.path.join(FLAGS.train_dir, "model.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0


def evaluate():
    with tf.Session() as sess:
        model = create_model(sess)
        print('model loaded')

        features = read_data('train_3_0.pickle')
        print('read data, contains {} sequences'.format(len(features)))

        batch_size = FLAGS.batch_size
        num_batch = len(features) // batch_size  # the number of batches
        total_loss = 0.0

        print('total steps for evaluation: {}'.format(num_batch))
        for i in range(num_batch):
            start_idx = i * batch_size
            encoder_inputs = model.get_batch(features, shuffle=False, start=start_idx)
            batch_loss = model.step(session=sess, inputs=encoder_inputs)
            total_loss += batch_loss

        loss = total_loss / num_batch  # get the average loss for each batch
        print('Evaluation completed, loss {}'.format(loss))


if __name__ == '__main__':
    evaluate()
