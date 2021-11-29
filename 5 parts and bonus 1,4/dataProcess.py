import numpy as np
from Loading_Datasets import get_data_sets


def ready_data():
    # part 1
    train_set, test_set = get_data_sets()
    x_train_set = []
    y_train_set = []
    x_test_set = []
    y_test_set = []
    for i in range(len(train_set)):
        x_train_set.append(train_set[i][0])
        y_train_set.append(train_set[i][1])
    for i in range(len(test_set)):
        x_test_set.append(test_set[i][0])
        y_test_set.append(test_set[i][1])
    x_train_set = np.array(x_train_set)
    x_train_set = x_train_set.reshape(x_train_set.shape[0], x_train_set.shape[1])
    y_train_set = np.array(y_train_set)
    y_train_set = y_train_set.reshape(y_train_set.shape[0], y_train_set.shape[1])
    x_test_set = np.array(x_test_set)
    x_test_set = x_test_set.reshape(x_test_set.shape[0], x_test_set.shape[1])
    y_test_set = np.array(y_test_set)
    y_test_set = y_test_set.reshape(y_test_set.shape[0], y_test_set.shape[1])
    return x_train_set.T, y_train_set.T, x_test_set.T, y_test_set.T
