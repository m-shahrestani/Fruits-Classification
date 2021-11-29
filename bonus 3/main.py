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
weights.append(np.random.normal(size=(150, 120)))
biases.append(np.zeros(150).reshape(150, 1))
grad_w.append(np.zeros(150 * 120).reshape(150, 120))
grad_b.append(np.zeros(150).reshape(150, 1))
# layer 2
weights.append(np.random.normal(size=(60, 150)))
biases.append(np.zeros(60).reshape(60, 1))
grad_w.append(np.zeros(60 * 150).reshape(60, 150))
grad_b.append(np.zeros(60).reshape(60, 1))
# layer 3
weights.append(np.random.normal(size=(6, 60)))
biases.append(np.zeros(6).reshape(6, 1))
grad_w.append(np.zeros(6 * 60).reshape(6, 60))
grad_b.append(np.zeros(6).reshape(6, 1))


def reinitialize_grad():
    grad_w[0] = np.zeros(150 * 120).reshape(150, 120)
    grad_b[0] = np.zeros(150).reshape(150, 1)
    grad_w[1] = np.zeros(60 * 150).reshape(60, 150)
    grad_b[1] = np.zeros(60).reshape(60, 1)
    grad_w[2] = np.zeros(6 * 60).reshape(6, 60)
    grad_b[2] = np.zeros(6).reshape(6, 1)


def shuffle(X, Y):
    order = np.random.permutation(X.shape[1])
    XX = np.zeros(X.shape)
    YY = np.zeros(Y.shape)
    for i in range(len(order)):
        XX[:, i] = X[:, order[i]]
        YY[:, i] = Y[:, order[i]]
    X = XX
    Y = YY


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def feedforward(x, y):
    layer1_neurons = sigmoid(weights[0] @ x + biases[0])
    layer2_neurons = sigmoid(weights[1] @ layer1_neurons + biases[1])
    layer3_neurons = sigmoid(weights[2] @ layer2_neurons + biases[2])
    nodes = {'L1': layer1_neurons, 'L2': layer2_neurons, 'L3': layer3_neurons}
    return np.argmax(layer3_neurons) == np.argmax(y), nodes


def get_accuracy(X, Y):
    correct = 0
    for i in range(X.shape[1]):
        c, _ = feedforward(X[:, i].reshape(X.shape[0], 1), Y[:, i].reshape(Y.shape[0], 1))
        correct += c
    return correct / X.shape[1]


def get_error(a, y):
    return np.sum(np.square(a - y))


def get_cost(X, Y):
    cost = 0
    for i in range(X.shape[1]):
        _, nodes = feedforward(X[:, i].reshape(X.shape[0], 1), Y[:, i].reshape(Y.shape[0], 1))
        c = get_error(nodes['L3'], Y[:, i].reshape(Y.shape[0], 1))
        cost += c
    return cost / X.shape[1]


def vectorized_SGD(X, Y, epoch, batch_size, show_plot=True):
    start_time = time.time()
    cost_array = []
    reinitialize_grad()
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


if __name__ == '__main__':
    print("bonus 3:\n", vectorized_SGD(x_train_set, y_train_set, 50, 10))
    accuracy = get_accuracy(x_train_set, y_train_set)
    print(" train set accuracy = ", accuracy * 100, "%")
    accuracy = get_accuracy(x_test_set, y_test_set)
    print(" test set accuracy = ", accuracy * 100, "%")