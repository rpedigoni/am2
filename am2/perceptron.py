# coding: utf-8
from neuron import Neuron, StaircaseTransferFunction


class Perceptron(object):
    def __init__(self, weights=[], learning_rate=0.1, function=None):
        self.weights = weights
        self.learning_rate = learning_rate
        self.function = function

    def create_neuron(self, weights=None):
        return Neuron(
            weights=weights or self.weights,
            transfer_function=StaircaseTransferFunction,
            function=self.function,
        )

    def run(self, inputs):
        assert self.weights, u'run .train() or set weight vector manually'

        return self.create_neuron().run(inputs)

    def train(self, dataset):
        """
        Using basic learning algorithm of:
        - http://en.wikipedia.org/wiki/Perceptron#Learning_algorithm
        """
        weights = [0.0 for i in dataset[0][0]]

        while True:
            error_count = 0
            for inputs, expected in dataset:
                result = self.create_neuron(weights=weights).run(inputs)
                if result != expected:
                    error = expected - result
                    error_count += 1

                    for index, value in enumerate(inputs):
                        weights[index] += self.learning_rate * error * value

            if not error_count:
                self.weights = weights
                return True
            # e se não encontrar? (loop infinito?)
