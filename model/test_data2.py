import sys
sys.path.insert(0, 'Libs')

import data_fetcher
import inputs_generator
from matplotlib import pyplot as plt
from neural_network_np import NeuralNetwork
import os
import numpy as np
import random

nn = NeuralNetwork(21, [1000, 500], 3, learning_rate=0.007)
#nn = NeuralNetwork(2, [100, 100], 1, learning_rate=0.07)


X_TRAINING = []
Y_TRAINING = []


for race in data_fetcher.data_races:
    for runner in race['participants']:
        Y_TRAINING.append(inputs_generator.runner_to_output(runner))
        X_TRAINING.append(inputs_generator.runner_to_input(runner, race))
"""

X_TRAINING = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
Y_TRAINING = [
    [0],
    [1],
    [1],
    [0]
]
"""
batch_size = 10000
iters = 1
epochs = 1000

def get_batch():
    i = random.randint(0, len(X_TRAINING) - batch_size)
    return (np.array(X_TRAINING[i:i+batch_size]), np.array(Y_TRAINING[i:i+batch_size]))

for i in range(epochs):
    (x, y) = get_batch()
    for iter in range(iters):
        for a in range(len(x)):
            nn.train(x[a], y[a])
    print("Epoch", i+1, "of", epochs, "cost:", round(nn.cost(X_TRAINING, Y_TRAINING), 3), "accuracy:", round(nn.accuracy(X_TRAINING, Y_TRAINING), 3), "%")
