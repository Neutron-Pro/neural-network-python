import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score


class NeuralNetwork:
    def __init__(self, dimensions, learningRate=0.1):
        self.activations = {}
        self.parameters = {}
        self.gradients = {}
        self.learningRate = learningRate
        self.lossHistory = []
        self.accuracyScoreHistory = []
        for c in range(1, len(dimensions)):
            self.parameters['w' + str(c)] = np.random.randn(dimensions[c], dimensions[c - 1])
            self.parameters['b' + str(c)] = np.random.randn(dimensions[c], 1)

    def forwardPropagation(self, data):
        self.activations = {'a0': data}

        for c in range(1, (len(self.parameters) // 2) + 1):
            Z = self.parameters['w' + str(c)].dot(self.activations['a' + str(c - 1)]) + self.parameters['b' + str(c)]
            self.activations['a' + str(c)] = 1 / (1 + np.exp(-Z))

    def backwardPropagation(self, results):
        m = results.shape[1]
        maxLength = len(self.parameters) // 2
        delta = self.activations['a' + str(maxLength)] - results
        self.gradients = {}

        for c in reversed(range(1, maxLength + 1)):
            self.gradients['dw' + str(c)] = 1 / m * np.dot(delta, self.activations['a' + str(c - 1)].T)
            self.gradients['db' + str(c)] = 1 / m * np.sum(delta, axis=1, keepdims=True)
            if c > 1:
                activation = self.activations['a' + str(c - 1)]
                delta = np.dot(self.parameters['w' + str(c)].T, delta) * activation * (1 - activation)

    def loss(self, result):
        lastActivation = self.activations['a' + str(len(self.parameters) // 2)]
        somme = np.sum(-result * np.log(lastActivation + 1e-15) - (1 - result) * np.log(1 - lastActivation + 1e-15))
        return 1 / len(result) * somme

    def update(self):
        for c in range(1, (len(self.parameters) // 2) + 1):
            self.parameters['w' + str(c)] -= self.learningRate * self.gradients['dw' + str(c)]

    def predicate(self, data):
        self.forwardPropagation(data)
        return self.activations['a' + str(len(self.parameters) // 2)]

    def learn(self, data, results, iteration=100, log=False):
        self.lossHistory = []
        self.accuracyScoreHistory = []

        for i in tqdm(range(iteration)):
            self.forwardPropagation(data)
            self.backwardPropagation(results)
            self.update()

            if log:
                self.lossHistory.append(self.loss(results))
                accuracy = accuracy_score(results.flatten(), (self.predicate(data) > 0.5).flatten())
                self.accuracyScoreHistory.append(accuracy)

