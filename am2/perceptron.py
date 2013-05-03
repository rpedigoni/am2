# coding: utf-8
from neuron import Neuron, StepTransferFunction


class Perceptron(object):
    def __init__(self, weights=[], learning_rate=0.005, function=None):
        self.weights = weights
        self.learning_rate = learning_rate
        self.function = function

    def create_neuron(self, weights=None):
        return Neuron(
            weights=weights or self.weights,
            transfer_function=StepTransferFunction,
            function=self.function,
        )

    def run(self, inputs):
        assert self.weights, u'run .train() or set weight vector manually'

        return self.create_neuron().run(inputs)

    def _hash_weights(self, weights):
        return '|'.join(map(str, weights))

    def _unhash_weights(self, hashed_weights):
        return map(float, hashed_weights.split('|'))

    def train(self, dataset, max_loops=10000):
        """
        Using basic learning algorithm of:
        - http://en.wikipedia.org/wiki/Perceptron#Learning_algorithm

        Update: added an array containing all weights and its error rate
        """
        weights = [0.0 for i in dataset[0][0]]
        weight_errors = {}

        i = 0
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
            else:
                weight_errors.update({self._hash_weights(weights): error_count})

            i += 1
            print i

            if i == max_loops:
                self.weights = self._unhash_weights(sorted(weight_errors, key=lambda k: weight_errors[k])[0])
                print 'error_count: {0}'.format(weight_errors[self._hash_weights(self.weights)])
                return True
