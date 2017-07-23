import pickle
import os
import numpy as np
import tensorflow as tf
from scipy.spatial import distance

from model import Seq2SeqModel

tf.app.flags.DEFINE_integer("steps_per_checkpoint", 400,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_string("train_dir", "./model-sum-day", "Training directory.")
tf.app.flags.DEFINE_integer('total_steps', 20000,
                            'How many training steps to take')
tf.app.flags.DEFINE_integer('batch_size', 64,
                            'the batch size for training')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
tf.app.flags.DEFINE_integer('time_len', 120, 'the length of time window (in minute)')
tf.app.flags.DEFINE_integer('data_size', 2, 'the dimension of each data point')
tf.app.flags.DEFINE_integer('unit_size', 64, 'the size of each LSTM layer (dimension of hidden states)')
tf.app.flags.DEFINE_integer('num_layers', 1, 'the number of LSTM layers in the model')

FLAGS = tf.app.flags.FLAGS


def read_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def create_model(session, feed_previous):
    model = Seq2SeqModel(data_size=FLAGS.data_size, time_len=FLAGS.time_len,
                         unit_size=FLAGS.unit_size, num_layers=FLAGS.num_layers, batch_size=FLAGS.batch_size,
                         learning_rate=FLAGS.learning_rate, feed_previous=feed_previous)

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
        model = create_model(sess, feed_previous=True)
        print('model created')

        features = read_data('train_summer_day.pickle')['feature']
        val_features = read_data('val_summer_day.pickle')['feature']

        if len(features[0].shape) == 1:  # if the data dimension is only 1, we expend the dimension by 1
            for i, element in enumerate(features):
                features[i] = np.expand_dims(element, axis=1)
            for i, element in enumerate(val_features):
                val_features[i] = np.expand_dims(element, axis=1)

        loss = 0.0
        current_step = 0

        for _ in range(FLAGS.total_steps):
            encoder_inputs, decoder_inputs = model.get_batch(features)
            step_loss = model.step(session=sess, encoder_inputs=encoder_inputs, decoder_inputs=decoder_inputs)
            loss += step_loss / FLAGS.steps_per_checkpoint

            current_step += 1

            if current_step % FLAGS.steps_per_checkpoint == 0:
                val_encoder_inputs, val_decoder_inputs = model.get_batch(val_features)
                val_loss = model.step(session=sess, encoder_inputs=val_encoder_inputs,
                                      decoder_inputs=val_decoder_inputs, train=False)

                print('global step {}, train loss {}, val loss {}'.format(model.global_step.eval(),
                                                                          loss, val_loss))
                checkpoint_path = os.path.join(FLAGS.train_dir, "model.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                loss = 0.0


def evaluate():
    with tf.Session() as sess:
        model = create_model(sess, feed_previous=False)
        print('model loaded')

        features = read_data('test1_summer_day.pickle')['feature']
        print('read data, contains {} sequences'.format(len(features)))

        if len(features[0].shape) == 1:  # if the data dimension is only 1, we expend the dimension by 1
            for i, element in enumerate(features):
                features[i] = np.expand_dims(element, axis=1)

        batch_size = FLAGS.batch_size
        num_batch = len(features) // batch_size  # the number of batches
        total_loss = 0.0

        print('total steps for evaluation: {}'.format(num_batch))
        for i in range(num_batch):
            start_idx = i * batch_size
            encoder_inputs, decoder_inputs = model.get_batch(features, shuffle=False, start=start_idx)
            batch_loss = model.step(session=sess, encoder_inputs=encoder_inputs,
                                    decoder_inputs=decoder_inputs, train=False)
            total_loss += batch_loss

        loss = total_loss / num_batch  # get the average loss for each batch
        print('Evaluation completed, loss {}'.format(loss))


def get_errs(read_path, out_name):
    with tf.Session() as sess:
        model = create_model(sess, feed_previous=False)
        print('model loaded')

        data = read_data(read_path)
        features = data['feature']

        if len(features[0].shape) == 1:  # if the data dimension is only 1, we expend the dimension by 1
            for i, element in enumerate(features):
                features[i] = np.expand_dims(element, axis=1)

        print('read data for getting err vecs, contains {} sequences'.format(len(features)))

        batch_size = FLAGS.batch_size
        num_batch = len(features) // batch_size  # the number of batches

        vectors = []
        for i in range(num_batch):
            start_idx = i * batch_size
            encoder_inputs, decoder_inputs = model.get_batch(features, shuffle=False, start=start_idx)

            batch_err_vec = model.get_err_vec(session=sess, encoder_inputs=encoder_inputs,
                                              decoder_inputs=decoder_inputs)
            vectors.append(batch_err_vec)

        vectors = np.concatenate(vectors, axis=0)
        result = {'time': data['time'], 'vectors': vectors, 'index': data['index']}

        with open(out_name + '.pickle', 'wb') as f:
            pickle.dump(result, f)


def fit_err_vec(vec_filename):
    with open(vec_filename, 'rb') as f:
        data = pickle.load(f)
        vecs = data['vectors']
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


def detect_anomaly(ref_file, test_vec_file, orig_test_file):
    means, covs = fit_err_vec(ref_file)

    with open(ref_file, 'rb') as f:
        data = pickle.load(f)
        ref_vecs = data['vectors']

    with open(test_vec_file, 'rb') as f:
        data = pickle.load(f)
        test_vecs = data['vectors']
        test_indices = data['index']

    anomalies = []
    for t in range(len(means)):

        # get the maximal distance in the reference set
        max_ref_dis = 0
        for i in range(len(ref_vecs)):

            if len(covs[t].shape) == 0:  # convert scalar to 1by1 mat
                covs[t] = np.linalg.inv(covs[t].reshape((1, 1)))

            dis = distance.mahalanobis(u=ref_vecs[i][t], v=means[t], VI=np.linalg.inv(covs[t]))
            if dis > max_ref_dis:
                max_ref_dis = dis

        # print('max distance for time {} is {}'.format(t, max_ref_dis))

        for i in range(len(test_vecs)):
            dis = distance.mahalanobis(u=test_vecs[i][t], v=means[t], VI=np.linalg.inv(covs[t]))
            if dis > max_ref_dis * 1.25:
                anomalies.append(([test_indices[i], t]))
                # print('anomaly detected, with distance {}'.format(dis))

    with open(orig_test_file, 'rb') as f:
        times = pickle.load(f)['time']

    anomaly_times, anomaly_indices = summarize_time(anomalies, times)
    return anomaly_times, anomaly_indices


def summarize_time(anomalies, times):
    anomaly_indices = []
    for anomaly in anomalies:
        start, shift = anomaly[0], anomaly[1]
        anomaly_indices.append(start + shift)
    anomaly_indices = sorted(list(set(anomaly_indices)))
    anomaly_times = []
    for index in anomaly_indices:
        anomaly_times.append(times[index])
    return anomaly_times, anomaly_indices


if __name__ == '__main__':
    '''
    For a random initialized model, the loss is larger than 1, almost always around 1.5 
    
    For 1 layer LSTM with size of 32, the result is relatively reasonable for 2-d feature vector
    
    Now for winter morning data, train_loss and val_loss is almost 0.05 while testing is 0.26
    
    The same standard only gets 80 anomaly points on training set. The result looks reasonable
    
    '''
    # train()
    # evaluate()
    # get_errs(read_path='val_summer_day.pickle', out_name='val_err_summer_day')

    # get_errs(read_path='test1_summer_day.pickle', out_name='test1_err_summer_day')

    # with open('testv1_err_vec.pickle', 'rb') as f:
    # data = pickle.load(f)
    # print(data['vectors'][0].shape)
    # print(len(data['vectors']))

    # times, indices = detect_anomaly(ref_file='val_err_summer_day.pickle', test_vec_file='test1_err_summer_day.pickle',
    #                                 orig_test_file='test1_0.pickle')
    # print(len(times))
    # for time in times:
    #     print(time)
