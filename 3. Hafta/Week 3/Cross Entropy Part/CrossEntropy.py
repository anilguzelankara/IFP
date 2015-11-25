import matplotlib.pylab as plt
import numpy as np


class Sigmoid(object):

    def activate(self, z):
        return 1. / (1. + np.exp(-z))

    def diff(self, z):
        return self.activate(z) * (1. - self.activate(z))


class QuadraticCost():

    def cost(self, y, out):
        return 0.5 * ((y - out) ** 2.)

    def diff(self, y, out):
        return (out - y)


class CrossEntropyCost(object):

    def cost(self, y, out):
        return -(y  * np.log(out) + (1. - y) * np.log(1. - out))

    def diff(self, y, out):
        return (y - out) / (out * (out - 1.))


def singleNeuronModel(weight, bias, x=1.0, y=0.0, costFunction=QuadraticCost(),eta=0.15, activationFunction=Sigmoid(),epoch=300):
    allCosts = []
    for i in range(epoch):
        z = x * weight+bias
        out = activationFunction.activate(z)
        weight -= eta * costFunction.diff(y, out) * activationFunction.diff(z) * out
        bias -= eta * costFunction.diff(y, out) * activationFunction.diff(z)
        allCosts.append(costFunction.cost(y, out))

    print "weight: {}, bias: {}, output: {}".format (weight, bias, allCosts[-1])
    plt.plot(range(epoch), allCosts)
    plt.xlabel('epochs')
    plt.ylabel('cost')
    plt.show()

#singleNeuronModel(weight=0.6, bias=0.9)

#singleNeuronModel(weight=2.0, bias=2.0)
singleNeuronModel(weight=2.0, bias=2.0, costFunction=CrossEntropyCost())
#singleNeuronModel(weight=0.6, bias=0.9, costFunction=CrossEntropyCost())
