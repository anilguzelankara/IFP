import matplotlib.pylab as plt
import numpy as np


class QuadraticCost():

    def cost(self, y, out):
        return 0.5 * ((y - out) ** 2.)

    def diff(self, y, out):
        return (out - y)

class Linear():

    def activate(self, z):
        return z

    def diff(self, z):
        return 1

class ReLU():

    def activate(self, z):
        return max(0, z)

    def diff(self, z):
        if z > 0:
            return 1
        else:
            return 0



def compareTrainCost(*args):

    costs = []
    for index, arg in enumerate(args):
        (x, y) = arg[0]  # input and expected output
        (weight, bias) = arg[1]  # initial weight and bias
        costFunction = arg[2]
        activationFunction = arg[3]
        eta = arg[4]  # learning rate
        llambda = arg[5] # regularization constant
        epoch = arg[6]
        costs.append([])
        for i in range(epoch):
            z = x * weight + bias
            out = activationFunction.activate(z)
            weight =  weight - eta * llambda * weight - eta * costFunction.diff(y, out) * activationFunction.diff(z) * out
            bias = bias - eta * costFunction.diff(y, out) * activationFunction.diff(z)
            costs[index].append(costFunction.cost(y, out))

        # Last out value with updated weights and biases.
        z = x * weight + bias
        out = activationFunction.activate(z)
        costs[index].append(costFunction.cost(y, out))

    # Draw train cost vs. epoch
    for index, cost in enumerate(costs):
        plt.plot(range(epoch + 1), cost, label="model {}".format(index + 1))
    plt.xlabel('epochs')
    plt.ylabel('cost')
    plt.legend()
    plt.show()


def compareWeightBias(*args):
    biases = []
    weights = []
    for index, arg in enumerate(args):
        (x, y) = arg[0]
        (weight, bias) = arg[1]
        costFunction = arg[2]
        activationFunction = arg[3]
        eta = arg[4]
        llambda = arg[5]
        epoch = arg[6]
        biases.append([])
        weights.append([])
        for  i in range(epoch):
            z = x * weight + bias
            out = activationFunction.activate(z)
            weight =  weight - eta * llambda * weight - eta * costFunction.diff(y, out) * activationFunction.diff(z) * out
            bias = bias - eta * costFunction.diff(y, out) * activationFunction.diff(z)
            biases[index].append(bias)
            weights[index].append(weight)

        for index, weight in enumerate(weights):
            plt.plot(range(epoch), weight, label="model {}".format(index + 1))
        plt.xlabel('epochs')
        plt.ylabel('weight')
        plt.legend()
        plt.show()

        for index, bias in enumerate(biases):
            plt.plot(range(epoch), bias, label="model {}".format(index + 1))
        plt.xlabel('epochs')
        plt.ylabel('bias')
        plt.legend()
        plt.show()


        #print len(weights[0])
        #print len(biases[0])


neuron1 = ((1.0, 0.0),
           (0.8, 1.3),
           (QuadraticCost()),
           (Linear()),
           0.15,
           0.0,
           30)

neuron2 = ((1.0, 0.0),
           (0.2, 1.3),
           (QuadraticCost()),
           (ReLU()),
           0.15,
           0.0,
           30)


#print compareTrainCost(neuron1, neuron2)

print compareWeightBias(neuron1, neuron2)
