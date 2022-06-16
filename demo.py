import numpy as np
from sklearn.datasets import make_circles
from neural_network import NeuralNetwork
import matplotlib.pyplot as plt


def makeDatasets():
    data1, results1 = make_circles(n_samples=50, noise=0.1, factor=0.3, random_state=0)
    data2, results2 = make_circles(n_samples=50, noise=0.1, factor=0.3, random_state=0)
    data1 += data1.min()
    data2 -= data2.min()

    return np.insert(data1, 0, data2, axis=0), np.insert(results1, 0, results2, axis=0)


data, results = makeDatasets()
for i in range(0, len(data)):
    data[i][0] = data[i][0] * -1

data2, results2 = makeDatasets()

data = np.insert(data, 0, data2, axis=0).T
results = np.insert(results, 0, results2, axis=0)

data = data.reshape((-1, data.shape[-1])) / data.max()
results = results.reshape((1, results.shape[0]))

dimensions = list((32, 32, 32))
dimensions.insert(0, data.shape[0])
dimensions.append(results.shape[0])
network = NeuralNetwork(dimensions)

print('Learning in progress...')
network.learn(data, results, iteration=100000, log=True)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 4))
ax[0].plot(network.lossHistory, label='Loss')
ax[0].legend()

ax[1].plot(network.accuracyScoreHistory, label='Accuracy')
ax[1].legend()
plt.show()

f, ax = plt.subplots()
ax.scatter(data[0, :], data[1, :], c=results, cmap='bwr', s=50)
xLimit = ax.get_xlim()
yLimit = ax.get_ylim()
resolution = 100
x = np.linspace(xLimit[0], xLimit[1], resolution)
y = np.linspace(yLimit[0], yLimit[1], resolution)

a1, a2 = np.meshgrid(x, y)
a = np.vstack((a1.ravel(), a2.ravel()))
p = (network.predicate(a) > 0.5).reshape((resolution, resolution))

ax.pcolormesh(a1, a2, p, cmap='bwr', alpha=0.2, zorder=-1)
plt.show()
