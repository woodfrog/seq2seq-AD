import pickle
import os
import numpy as np
import tensorflow as tf
from scipy.spatial import distance

from model import Seq2SeqModel

tf.app.flags.DEFINE_integer("steps_per_checkpoint", 400,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_string("train_dir", "./train", "Training directory.")
tf.app.flags.DEFINE_integer('total_steps', 10000,
                            'How many training steps to take')
tf.app.flags.DEFINE_integer('batch_size', 64,
                            'the batch size for training')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
tf.app.flags.DEFINE_integer('time_len', 60, 'the length of time window (in minute)')
tf.app.flags.DEFINE_integer('data_size', 6, 'the dimension of each data point')
tf.app.flags.DEFINE_integer('unit_size', 64, 'the size of each LSTM layer (dimension of hidden states)')
tf.app.flags.DEFINE_integer('num_layers', 1, 'the number of LSTM layers in the model')

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
        loss = 0.0
        current_step = 0

        for _ in range(FLAGS.total_steps):
            encoder_inputs = model.get_batch(features)
            step_loss = model.step(session=sess, inputs=encoder_inputs)
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            if current_step % FLAGS.steps_per_checkpoint == 0:
                val_features = read_data('train_0_1.pickle')
                val_inputs = model.get_batch(val_features)
                val_loss = model.step(session=sess, inputs=val_inputs, train=False)

                print('global step {}, train loss {}, val loss {}'.format(model.global_step.eval(),
                                                                          loss, val_loss))
                checkpoint_path = os.path.join(FLAGS.train_dir, "model.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                loss = 0.0


def evaluate():
    with tf.Session() as sess:
        model = create_model(sess)
        print('model loaded')

        features = read_data('train_0_0.pickle')
        print('read data, contains {} sequences'.format(len(features)))

        batch_size = FLAGS.batch_size
        num_batch = len(features) // batch_size  # the number of batches
        total_loss = 0.0

        print('total steps for evaluation: {}'.format(num_batch))
        for i in range(num_batch):
            start_idx = i * batch_size
            encoder_inputs = model.get_batch(features, shuffle=False, start=start_idx)
            batch_loss = model.step(session=sess, inputs=encoder_inputs, train=False)
            total_loss += batch_loss

        loss = total_loss / num_batch  # get the average loss for each batch
        print('Evaluation completed, loss {}'.format(loss))


def get_errs(out_name):
    with tf.Session() as sess:
        model = create_model(sess)
        print('model loaded')

        features = read_data('train_3_0.pickle')
        print('read data, contains {} sequences'.format(len(features)))

        batch_size = FLAGS.batch_size
        num_batch = len(features) // batch_size  # the number of batches

        vectors = []
        for i in range(num_batch):
            start_idx = i * batch_size
            encoder_inputs = model.get_batch(features, shuffle=False, start=start_idx)
            batch_err_vec = model.get_err_vec(session=sess, inputs=encoder_inputs)
            vectors.append(batch_err_vec)

        vectors = np.concatenate(vectors, axis=0)

        with open(out_name + '.pickle', 'wb') as f:
            pickle.dump(vectors, f)


def fit_err_vec(vecs):
    vecs_per_t = np.split(vecs, indices_or_sections=vecs.shape[1], axis=1)
    for i in range(len(vecs_per_t)):
        vecs_per_t[i] = np.squeeze(vecs_per_t[i], axis=1)
    means = []
    covs = []
    for t, err_vecs in enumerate(vecs_per_t):
        mean = np.mean(err_vecs, axis=0)
        cov = np.cov(err_vecs, rowvar=False)
        means.append(mean)
        covs.append(cov)
    return means, covs


if __name__ == '__main__':
    '''
    For a random initialized model, the loss is larger than 1, almost always around 1.5 
    
    For 1 layer LSTM with size of 64, 
    
    '''
    evaluate()

    # vecs = None
    # with open('err_vec_0.pickle', 'rb') as f:
    #     vecs = pickle.load(f)
    #
    # means, covs = fit_err_vec(vecs)
    #
    # total_diff = 0
    # print(means[30])
    # for i in range(len(vecs)):
    #     total_diff += distance.mahalanobis(u=vecs[i][30], v=means[30], VI=np.linalg.inv(covs[30]))
    # print(total_diff / len(vecs))
