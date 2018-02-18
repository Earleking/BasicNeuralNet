import numpy as np
import random
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class net:
    def __init__(self, hLayers, size):
        x = hLayers + 2
        self.activations = [[0 for i in range(0)]for l in range(x)]
        self.errors = [[0 for i in range(0)]for l in range(x)]
        self.weights = [[0 for k in range(0)]for l in range(x)]
        self.biases = [[0 for k in range(0)]for l in range(x)]
        for l in range(0, x):
            for j in range(size[l]):
                self.activations[l].append(0)
                self.errors[l].append(0)
                self.weights[l].append([])
                self.biases[l].append([])
                if(l != 0):
                    for k in range(size[l - 1]):
                        self.weights[l][j].append(random.random() * 5 - 2.5)
                        self.biases[l][j].append(0)
        #print(self.activations[0])
    def activate(self, inputs):
        #set input activations
        for j in range (len(inputs)):
            self.activations[0][j] = inputs[j]
        for l in range (1, len(self.weights)):
            for j in range(len(self.activations[l])):
                self.activations[l][j] = sigmoid(np.dot(self.weights[l][j], self.activations[l - 1]))
        return self.activations[len(self.activations) - 1]
    def learn(self, inputs, outputs):
        #output errors
        for i in range(len(inputs)):
            #output layer
            outs = self.activate(inputs[i])
            error = np.subtract(outs, outputs[i])
            devSig = np.multiply(outs, np.subtract(1, outs))
            error = np.multiply(error, devSig)
            self.errors[len(self.   activations) - 1] = error
            #rest of layers
            for l in reversed(range(1, len(self.activations) - 1)):
                weightT = np.transpose(self.weights[l + 1])
                weightT = np.matmul(weightT, self.errors[l + 1])
                devSig = np.multiply(self.activations[l], np.subtract(1, self.activations[l]))
                error = np.multiply(weightT, devSig)
                self.errors[l] = error

            #apply errors: This should be optimized in matrices later
            for l in range(1, len(self.activations)):
                for j in range(len(self.weights[l])):
                    for k in range(len(self.weights[l][j])):
                        self.weights[l][j][k] -= (self.errors[l][j] * self.activations[l - 1][k]) * 10

a = net(2, [2, 5, 4, 2])
print(a.activate([.5, .6]))
ins = [[.5, .6],[.5, .6],[.5, .6],[.5, .6],[.5, .6],[.5, .6],[.5, .6],[.5, .6],[.5, .6],[.5, .6],[.5, .6],[.5, .6]]
outs = [[.1, .4],[.1, .4],[.1, .4],[.1, .4],[.1, .4],[.1, .4],[.1, .4],[.1, .4],[.1, .4],[.1, .4],[.1, .4],[.1, .4]]
a.learn(ins, outs)
a.learn(ins, outs)
print(a.activate([.5, .6]))
