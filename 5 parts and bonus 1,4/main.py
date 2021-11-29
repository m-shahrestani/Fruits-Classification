import time
import numpy as np
import matplotlib.pyplot as plt

from dataProcess import ready_data

x_train_set, y_train_set, x_test_set, y_test_set = ready_data()
learning_rate = 1
accuracy = 0
weights = []
biases = []
grad_w = []
grad_b = []
# layer 1
weights.append(np.random.normal(size=(150, 102)))
biases.append(np.zeros(150).reshape(150, 1))
grad_w.append(np.zeros(150 * 102).reshape(150, 102))
grad_b.append(np.zeros(150).reshape(150, 1))
# layer 2
weights.append(np.random.normal(size=(60, 150)))
biases.append(np.zeros(60).reshape(60, 1))
grad_w.append(np.zeros(60 * 150).reshape(60, 150))
grad_b.append(np.zeros(60).reshape(60, 1))
# layer 3
weights.append(np.random.normal(size=(4, 60)))
biases.append(np.zeros(4).reshape(4, 1))
grad_w.append(np.zeros(4 * 60).reshape(4, 60))
grad_b.append(np.zeros(4).reshape(4, 1))


def reinitialize_parameters():
    weights[0] = np.random.normal(size=(150, 102))
    biases[0] = np.zeros(150).reshape(150, 1)
    weights[1] = np.random.normal(size=(60, 150))
    biases[1] = np.zeros(60).reshape(60, 1)
    weights[2] = np.random.normal(size=(4, 60))
    biases[2] = np.zeros(4).reshape(4, 1)


def reinitialize_grad():
    grad_w[0] = np.zeros(150 * 102).reshape(150, 102)
    grad_b[0] = np.zeros(150).reshape(150, 1)
    grad_w[1] = np.zeros(60 * 150).reshape(60, 150)
    grad_b[1] = np.zeros(60).reshape(60, 1)
    grad_w[2] = np.zeros(4 * 60).reshape(4, 60)
    grad_b[2] = np.zeros(4).reshape(4, 1)


def shuffle(X, Y):
    order = np.random.permutation(X.shape[1])
    XX = np.zeros(X.shape)
    YY = np.zeros(Y.shape)
    for i in range(len(order)):
        XX[:, i] = X[:, order[i]]
        YY[:, i] = Y[:, order[i]]
    X = XX
    Y = YY


# part 2
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# bonus-4
def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()


# part 2
def feedforward(x, y):
    layer1_neurons = sigmoid(weights[0] @ x + biases[0])
    layer2_neurons = sigmoid(weights[1] @ layer1_neurons + biases[1])
    layer3_neurons = sigmoid(weights[2] @ layer2_neurons + biases[2])
    nodes = {'L1': layer1_neurons, 'L2': layer2_neurons, 'L3': layer3_neurons}
    return np.argmax(layer3_neurons) == np.argmax(y), nodes


# bonus 4
def feedforward_softmax(x, y):
    layer1_neurons = sigmoid(weights[0] @ x + biases[0])
    layer2_neurons = sigmoid(weights[1] @ layer1_neurons + biases[1])
    layer3_neurons = softmax(weights[2] @ layer2_neurons + biases[2])
    nodes = {'L1': layer1_neurons, 'L2': layer2_neurons, 'L3': layer3_neurons}
    return np.argmax(layer3_neurons) == np.argmax(y), nodes


# part 2
def get_accuracy_softmax(X, Y):
    correct = 0
    for i in range(X.shape[1]):
        c, _ = feedforward_softmax(X[:, i].reshape(X.shape[0], 1), Y[:, i].reshape(Y.shape[0], 1))
        correct += c
    return correct / X.shape[1]


# part 2
def get_accuracy(X, Y):
    correct = 0
    for i in range(X.shape[1]):
        c, _ = feedforward(X[:, i].reshape(X.shape[0], 1), Y[:, i].reshape(Y.shape[0], 1))
        correct += c
    return correct / X.shape[1]


# part 3
def get_error(a, y):
    return np.sum(np.square(a - y))


# part 3
def get_cost(X, Y):
    cost = 0
    for i in range(X.shape[1]):
        _, nodes = feedforward(X[:, i].reshape(X.shape[0], 1), Y[:, i].reshape(Y.shape[0], 1))
        c = get_error(nodes['L3'], Y[:, i].reshape(Y.shape[0], 1))
        cost += c
    return cost / X.shape[1]


# part 3
def SGD(X, Y, epoch, batch_size, show_plot=True, initparam=True):
    if initparam:
        reinitialize_parameters()
    start_time = time.time()
    cost_array = []
    for ep in range(epoch):
        shuffle(X, Y)
        for index in range(0, X.shape[1], batch_size):
            batch_x = X[:, index * batch_size: min(X.shape[1], (index + 1) * batch_size)]
            batch_y = Y[:, index * batch_size: min(Y.shape[1], (index + 1) * batch_size)]
            reinitialize_grad()
            for img in range(batch_x.shape[1]):
                inputs = batch_x[:, img].reshape(batch_x.shape[0], 1)
                labels = batch_y[:, img].reshape(batch_y.shape[0], 1)
                _, nodes = feedforward(inputs, labels)
                derv_a_layer2 = np.zeros(60)
                derv_a_layer1 = np.zeros(150)
                for j in range(4):
                    for k in range(60):
                        grad_w[2][j, k] += 2 * (nodes['L3'][j, 0] - labels[j, 0]) * nodes['L3'][j, 0] * (
                                1 - nodes['L3'][j, 0]) * nodes['L2'][k, 0]
                    grad_b[2][j, 0] += 2 * (nodes['L3'][j, 0] - labels[j, 0]) * nodes['L3'][j, 0] * (
                            1 - nodes['L3'][j, 0])
                for k in range(60):
                    for jj in range(4):
                        derv_a_layer2[k] += 2 * (nodes['L3'][jj, 0] - labels[jj, 0]) * nodes['L3'][jj, 0] * (
                                1 - nodes['L3'][jj, 0]) * weights[2][jj, k]
                for k in range(60):
                    for m in range(150):
                        derv_ak = derv_a_layer2[k]
                        grad_w[1][k, m] += derv_ak * nodes['L2'][k, 0] * (1 - nodes['L2'][k, 0]) * nodes['L1'][m, 0]
                    grad_b[1][k, 0] += derv_ak * nodes['L2'][k, 0] * (1 - nodes['L2'][k, 0])
                for m in range(150):
                    for kk in range(60):
                        derv_a_layer1[m] += derv_a_layer2[kk] * nodes['L2'][kk, 0] * (1 - nodes['L2'][kk, 0]) * \
                                            weights[1][kk, m]
                for m in range(150):
                    for v in range(102):
                        derv_am = derv_a_layer1[m]
                        grad_w[0][m, v] += derv_am * nodes['L1'][m, 0] * (1 - nodes['L1'][m, 0]) * inputs[v, 0]
                    grad_b[0][m, 0] += derv_am * nodes['L1'][m, 0] * (1 - nodes['L1'][m, 0])
            weights[2] = weights[2] - learning_rate * (
                    grad_w[2] / (min(X.shape[1], (index + 1) * batch_size) - index * batch_size))
            weights[1] = weights[1] - learning_rate * (
                    grad_w[1] / (min(X.shape[1], (index + 1) * batch_size) - index * batch_size))
            weights[0] = weights[0] - learning_rate * (
                    grad_w[0] / (min(X.shape[1], (index + 1) * batch_size) - index * batch_size))
            biases[2] = biases[2] - learning_rate * (
                    grad_b[2] / (min(X.shape[1], (index + 1) * batch_size) - index * batch_size))
            biases[1] = biases[1] - learning_rate * (
                    grad_b[1] / (min(X.shape[1], (index + 1) * batch_size) - index * batch_size))
            biases[0] = biases[0] - learning_rate * (
                    grad_b[0] / (min(X.shape[1], (index + 1) * batch_size) - index * batch_size))
        cost = get_cost(X, Y)
        cost_array.append(cost)
    if show_plot is True:
        plt.plot([i for i in range(epoch)], cost_array)
        plt.show()
    return "epoch = " + str(epoch) + ", batch_size = " + str(batch_size) + ", Execution time(S) = " + str(
        (time.time() - start_time))


# part 4
def vectorized_SGD(X, Y, epoch, batch_size, show_plot=True, initparam=True):
    if initparam:
        reinitialize_parameters()
    start_time = time.time()
    cost_array = []
    for ep in range(epoch):
        shuffle(X, Y)
        for index in range(0, X.shape[1], batch_size):
            batch_x = X[:, index * batch_size: min(X.shape[1], (index + 1) * batch_size)]
            batch_y = Y[:, index * batch_size: min(Y.shape[1], (index + 1) * batch_size)]
            reinitialize_grad()
            for img in range(batch_x.shape[1]):
                inputs = batch_x[:, img].reshape(batch_x.shape[0], 1)
                labels = batch_y[:, img].reshape(batch_y.shape[0], 1)
                _, nodes = feedforward(inputs, labels)
                # layer 3
                grad_w[2] += ((2 * (nodes['L3'] - labels)) * nodes['L3'] * (1 - nodes['L3'])) @ np.transpose(
                    nodes['L2'])
                grad_b[2] += ((2 * (nodes['L3'] - labels)) * nodes['L3'] * (1 - nodes['L3']))
                grad_a2 = np.transpose(weights[2]) @ ((2 * (nodes['L3'] - labels)) * nodes['L3'] * (1 - nodes['L3']))
                # layer 2
                grad_w[1] += grad_a2 * nodes['L2'] * (1 - nodes['L2']) @ np.transpose(nodes['L1'])
                grad_b[1] += grad_a2 * nodes['L2'] * (1 - nodes['L2'])
                grad_a1 = np.transpose(weights[1]) @ (grad_a2 * nodes['L2'] * (1 - nodes['L2']))
                # layer 1
                grad_w[0] += grad_a1 * nodes['L1'] * (1 - nodes['L1']) @ np.transpose(inputs)
                grad_b[0] += grad_a1 * nodes['L1'] * (1 - nodes['L1'])
            weights[2] = weights[2] - learning_rate * (grad_w[2] / batch_size)
            weights[1] = weights[1] - learning_rate * (grad_w[1] / batch_size)
            weights[0] = weights[0] - learning_rate * (grad_w[0] / batch_size)
            biases[2] = biases[2] - learning_rate * (grad_b[2] / batch_size)
            biases[1] = biases[1] - learning_rate * (grad_b[1] / batch_size)
            biases[0] = biases[0] - learning_rate * (grad_b[0] / batch_size)
        cost = get_cost(X, Y)
        cost_array.append(cost)
    if show_plot is True:
        plt.plot([i for i in range(epoch)], cost_array)
        plt.show()
    return "epoch = " + str(epoch) + ", batch_size = " + str(batch_size) + ", Execution time(S) = " + str(
        (time.time() - start_time))


# bonus 4
def vectorized_SGD_softmax(X, Y, epoch, batch_size, show_plot=True, initparam=True):
    if initparam:
        reinitialize_parameters()
    start_time = time.time()
    cost_array = []
    for ep in range(epoch):
        shuffle(X, Y)
        for index in range(0, X.shape[1], batch_size):
            batch_x = X[:, index * batch_size: min(X.shape[1], (index + 1) * batch_size)]
            batch_y = Y[:, index * batch_size: min(Y.shape[1], (index + 1) * batch_size)]
            reinitialize_grad()
            for img in range(batch_x.shape[1]):
                inputs = batch_x[:, img].reshape(batch_x.shape[0], 1)
                labels = batch_y[:, img].reshape(batch_y.shape[0], 1)
                _, nodes = feedforward_softmax(inputs, labels)
                # layer 3
                grad_w[2] += ((2 * (nodes['L3'] - labels)) * nodes['L3'] * (1 - nodes['L3'])) @ np.transpose(
                    nodes['L2'])
                grad_b[2] += ((2 * (nodes['L3'] - labels)) * nodes['L3'] * (1 - nodes['L3']))
                grad_a2 = np.transpose(weights[2]) @ ((2 * (nodes['L3'] - labels)) * nodes['L3'] * (1 - nodes['L3']))
                # layer 2
                grad_w[1] += grad_a2 * nodes['L2'] * (1 - nodes['L2']) @ np.transpose(nodes['L1'])
                grad_b[1] += grad_a2 * nodes['L2'] * (1 - nodes['L2'])
                grad_a1 = np.transpose(weights[1]) @ (grad_a2 * nodes['L2'] * (1 - nodes['L2']))
                # layer 1
                grad_w[0] += grad_a1 * nodes['L1'] * (1 - nodes['L1']) @ np.transpose(inputs)
                grad_b[0] += grad_a1 * nodes['L1'] * (1 - nodes['L1'])
            weights[2] = weights[2] - learning_rate * (grad_w[2] / batch_size)
            weights[1] = weights[1] - learning_rate * (grad_w[1] / batch_size)
            weights[0] = weights[0] - learning_rate * (grad_w[0] / batch_size)
            biases[2] = biases[2] - learning_rate * (grad_b[2] / batch_size)
            biases[1] = biases[1] - learning_rate * (grad_b[1] / batch_size)
            biases[0] = biases[0] - learning_rate * (grad_b[0] / batch_size)
        cost = get_cost(X, Y)
        cost_array.append(cost)
    if show_plot is True:
        plt.plot([i for i in range(epoch)], cost_array)
        plt.show()
    return "epoch = " + str(epoch) + ", batch_size = " + str(batch_size) + ", Execution time(S) = " + str(
        (time.time() - start_time))


if __name__ == '__main__':
    # part 1
    print("part 1:\n", x_train_set.shape, y_train_set.shape, x_test_set.shape, y_test_set.shape)

    # part 2
    accuracy = 0
    for i in range(10):  # average in 10 tries
        accuracy += get_accuracy(x_train_set[0:200], y_train_set[0:200])
    print("\npart 2:\n average accuracy in 10 tries = ", (accuracy / 10) * 100, "%")

    # part 3
    print("\npart 3:\n", SGD(x_train_set[0:200], y_train_set[0:200], 5, 10, True, False), end='')  # epoch = 5
    accuracy = get_accuracy(x_train_set[0:200], y_train_set[0:200])
    print(", accuracy = ", accuracy * 100, "%")

    # part 4
    print("\npart 4-1:\n", vectorized_SGD(x_train_set[0:200], y_train_set[0:200], 10, 10), end='')  # epoch = 10
    accuracy = get_accuracy(x_train_set[0:200], y_train_set[0:200])
    print(", accuracy = ", accuracy * 100, "%")
    accuracy = 0
    for i in range(10):  # average between 10 tries
        vectorized_SGD(x_train_set[0:200], y_train_set[0:200], 20, 10, False)  # epoch = 20
        accuracy += get_accuracy(x_train_set[0:200], y_train_set[0:200])
    print("\npart 4-2:\n epoch = 20, batch_size = 10, average accuracy in 10 tries = ", (accuracy / 10) * 100, "%")

    # part 5
    vectorized_SGD(x_train_set, y_train_set, 20, 10, True, False)
    accuracy = get_accuracy(x_train_set, y_train_set)
    print("\npart 5-1:\n accuracy of train set = ", accuracy * 100, "%")
    accuracy = get_accuracy(x_test_set, y_test_set)
    print("\npart 5-2:\n accuracy of test set = ", accuracy * 100, "%")

    # bonus 1
    reinitialize_parameters()
    backup_weights = weights
    backup_biases = biases
    learning_rates = [0.01, 0.1, 0.9, 1, 1.1, 2, 10]
    epochs = [1, 2]
    batch_sizes = [5, 10]
    print("\nbonus 1:")
    for i in range(len(learning_rates)):
        for j in range(len(epochs)):
            for k in range(len(batch_sizes)):
                weights = backup_weights
                biases = backup_biases
                learning_rate = learning_rates[i]
                print(" learning rate = "+str(learning_rates[i])+",", vectorized_SGD(x_train_set, y_train_set, epochs[j], batch_sizes[k], False, False), end='')
                accuracy = get_accuracy(x_train_set, y_train_set)
                print(", accuracy = ", accuracy * 100, "%")

    # bonus 4
    accuracy1 = 0
    accuracy2 = 0
    for i in range(10):  # average between 10 tries
        reinitialize_parameters()
        backup_weights = weights
        backup_biases = biases
        vectorized_SGD(x_train_set, y_train_set, 10, 10, False, False)
        accuracy1 += get_accuracy(x_train_set, y_train_set)
        weights = backup_weights
        biases = backup_biases
        vectorized_SGD_softmax(x_train_set, y_train_set, 10, 10, False, False)
        accuracy2 += get_accuracy_softmax(x_train_set, y_train_set)
    print("\nbonus 4(with out softmax):\n epoch = 10, batch_size = 10, average accuracy in 10 tries = ",
          (accuracy1/10) * 100, "%")
    print("\nbonus 4(with softmax):\n epoch = 10, batch_size = 10, average accuracy in 10 tries = ",
          (accuracy2/10) * 100, "%")
