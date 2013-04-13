# coding: utf-8
import math


class BaseTransferFunction(object):
    def __init__(self, *args, **kwargs):
        pass

    def run(self, t):
        raise NotImplementedError()


class SigmoidTransferFunction(BaseTransferFunction):
    def run(self, t):
        return 1.0 / (1.0 + math.exp(1) ** (-1 * t))


class StaircaseTransferFunction(BaseTransferFunction):
    def __init__(self, function=None):
        if function:
            self.function = function
        else:
            self.function = lambda y: y >= 1

    def run(self, t):
        return 1 if self.function(t) else 0


class Neuron(object):
    def __init__(self, transfer_function, weights, *args, **kwargs):
        self.weights = weights
        self.transfer_function = transfer_function(*args, **kwargs)

    def integrator(self, inputs):
        r = 0.0

        for index, weight in enumerate(self.weights):
            r += inputs[index] * weight
        return r

    def run(self, inputs):
        return self.transfer_function.run(self.integrator(inputs))
