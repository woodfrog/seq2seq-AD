import time
import pickle
import os
import tensorflow as tf

from model import Seq2SeqModel
import data_utils

tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_string("train_dir", "./train", "Training directory.")

FLAGS = tf.app.flags.FLAGS


def read_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    features = data['feature']
    return features


def create_model(session):
    model = Seq2SeqModel(data_size=6, time_len=60,
                         unit_size=100, num_layers=2, batch_size=32, learning_rate=0.01)
    session.run(tf.global_variables_initializer())
    return model


def train():
    with tf.Session() as sess:
        model = create_model(sess)
        print('model created')

        features = read_data('train_0_0.pickle')
        start_time = time.time()
        step_time, loss = 0.0, 0.0
        current_step = 0

        while True:
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


if __name__ == '__main__':
    train()
