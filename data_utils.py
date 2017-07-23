import csv
import os
import pickle
import numpy as np
import re

from sklearn import preprocessing


def preprocess(file_path, division_ratios=(0.8, 0.2)):
    """
    1. To replace NA values with average of neighbouring data
    2. Standardization
    3. Separate the data set according to the given division ration
    :param file_path: the input file path
    :param division_ratios: the ratio for separating the data
    :return: No return value
    """
    data = []
    with open(file_path, mode='r') as f:
        reader = csv.reader(f)
        fieldnames = next(reader)
        for i, row in enumerate(reader):
            data.append(row)

    time = []
    feature = []
    # dealing with NA values
    for row_idx, row in enumerate(data):
        time.append(row[0] + ' ' + row[1])  # concatenate date and time
        for i in range(2, len(row)):  # start from the first feature
            if row[i] == 'NA':
                if row_idx < len(data) // 2:  # seek neighbours forwards
                    count = 0
                    total = 0
                    for neighbor in data[row_idx + 1:]:
                        if neighbor[i] != 'NA':
                            total += float(neighbor[i])
                            count += 1
                        if count == 5:
                            break
                    data[row_idx][i] = total * 1.0 / count
                else:  # seek neighbours backwards
                    count = 0
                    total = 0
                    for neighbor in data[row_idx - 1::-1]:
                        if neighbor[i] != 'NA':
                            total += float(neighbor[i])
                            count += 1
                        if count == 5:
                            break
                    data[row_idx][i] = total * 1.0 / count
            else:
                data[row_idx][i] = float(row[i])

        feature.append(np.asarray((data[row_idx][2],) + (data[row_idx][5],)))

    # do standardization, convert to zero mean, unit variance
    feature = preprocessing.scale(feature, axis=0)

    name, ext = os.path.splitext(file_path)

    # save separated data into pickle files
    # To separate the training data into different subsets
    start = 0
    for i, ratio in enumerate(division_ratios):
        end = int(start + len(time) * ratio)
        t_data, f_data = time[start:end], feature[start:end]
        out_path = name + '_{}'.format(i) + '.pickle'
        out_dict = {'fieldnames': fieldnames, 'time': t_data, 'feature': f_data}
        with open(out_path, 'wb') as f:
            pickle.dump(out_dict, f)
        start = end


def data_extract(data, tw_len, month_range, hour_range, out_path):
    times, features = data['time'], data['feature']
    new_times = []
    new_features = []
    indices = []

    i = 0

    while True:
        time = times[i]
        if i + tw_len * 60 > len(features):  # no enough data for building sequence of tw_len*60
            break
        month, hour_from, minute = parse_datetime(time)
        hour_to = hour_from + tw_len + (1 if minute > 0 else 0)
        if month in month_range and hour_from in hour_range and hour_to in hour_range:
            sequence = np.stack(features[i:i + tw_len * 60], axis=0)
            new_features.append(sequence)
            new_times.append(time)
            indices.append(i)
        i += 60

    new_data = {'time': new_times, 'feature': new_features, 'index': indices}
    with open(out_path + '.pickle', 'wb') as f:
        pickle.dump(new_data, f)


def parse_datetime(datetime):
    elements = re.findall(r'\d+', datetime)
    month = int(elements[1])
    hour = int(elements[3])
    minute = int(elements[4])
    return month, hour, minute


if __name__ == '__main__':
    # with open('train_3.pickle', 'rb') as f:
    #     data = pickle.load(f)
    #     times = data['time']
    #     count = 0
    #     for time in times:
    #         if 6 <= parse_datetime(time[1])[0] <= 8 and 18 <= parse_datetime(time[1])[1] <= 21:
    #             count += 1
    #     print(count)

    # with open('test_v1_0_0.pickle', 'rb') as f:
    #     data = pickle.load(f)
    # features = data['feature']
    # print(len(features))
    # print(data['time'])
    # for feature in features:
    # print(feature.shape)

    # preprocess('train.txt')
    # preprocess('test1.txt', division_ratios=(1,))
    #
    hour_range = tuple([i for i in range(6, 13)])

    with open('train_0.pickle', 'rb') as f:
        data = pickle.load(f)
        print(len(data['feature']))
        print(data['time'][0])
        data_extract(data, tw_len=2, month_range=(4, 5, 6, 7, 8, 9), hour_range=hour_range,
                     out_path='train_summer_day')

    with open('test1_0.pickle', 'rb') as f:
        data = pickle.load(f)
        print(len(data['feature']))
        print(data['time'][0])
        data_extract(data, tw_len=2, month_range=(4, 5, 6, 7, 8, 9), hour_range=hour_range,
                     out_path='test1_summer_day')

    with open('train_1.pickle', 'rb') as f:
        data = pickle.load(f)
        print(len(data['feature']))
        print(data['time'][0])
        data_extract(data, tw_len=2, month_range=(4, 5, 6, 7, 8, 9), hour_range=hour_range,
                     out_path='val_summer_day')
