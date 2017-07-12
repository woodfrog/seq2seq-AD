import csv
import os
import pickle


def preprocess(file_path):
    """
    To replace NA values with average of neighbouring data
    :param filename: The path for the file to be processed 
    :return: 
    """
    data = []
    fieldnames = None
    with open(file_path, mode='r') as f:
        reader = csv.reader(f)
        fieldnames = next(reader)
        for i, row in enumerate(reader):
            data.append(row)
            if i == 100:
                break

    # dealing with NA values
    for row_idx, row in enumerate(data):
        for i, entry in enumerate(row):
            if entry == 'NA':
                if row_idx < len(data) // 2:  # seek neighbours forwards
                    count = 0
                    total = 0
                    for neighbor in data[row_idx + 1:]:
                        if neighbor[i] != 'NA':
                            total += float(neighbor[i])
                            count += 1
                        if count == 5:
                            break
                    data[row_idx][i] = str(total * 1.0 / count)
                else:  # seek neighbours backwards
                    count = 0
                    total = 0
                    for neighbor in data[row_idx - 1::-1]:
                        if neighbor[i] != 'NA':
                            total += float(neighbor[i])
                            count += 1
                        if count == 5:
                            break
                    data[row_idx][i] = str(total * 1.0 / count)

    name, ext = os.path.splitext(file_path)

    # save as pickle file
    out_path = name + '_processed' + '.pickle'
    out_dict = {'fieldnames': fieldnames, 'data': data}
    pickle.dump(out_dict, out_path)


if __name__ == '__main__':
    preprocess('train.csv')
