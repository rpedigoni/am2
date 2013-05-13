# coding: utf-8
from neuron import Neuron, StepTransferFunction
from util import plot_dataset, plot_line, count_errors as util_count_errors


class Perceptron(object):
    def __init__(self, weights=[], learning_rate=0.005, function=None):
        self.weights = weights
        self.learning_rate = learning_rate
        self.function = function
        self.min_error = {}

    def create_neuron(self, weights=None):
        return Neuron(
            weights=weights or self.weights,
            transfer_function=StepTransferFunction,
            function=self.function,
        )

    def run(self, inputs):
        assert self.weights, u'run .train() or set weight vector manually'

        # add bias input (always 1)
        inputs = (1, ) + inputs

        return self.create_neuron().run(inputs)

    def count_errors(self, dataset):
        assert self.weights, u'run .train() or set weight vector manually'

        for i, row in enumerate(dataset):
            dataset[i] = ((1, ) + row[0], row[1])

        util_count_errors(
            self.create_neuron(weights=self.weights),
            dataset,
        )

    def plot(self, dataset):
        plot_dataset(dataset)

        if self.weights:
            plot_line(dataset, self.weights)

    def train(self, dataset, max_loops=1000):
        """
        Using basic learning algorithm of:
        - http://en.wikipedia.org/wiki/Perceptron#Learning_algorithm

        Update: added an array containing all weights and its error rate
        """

        dataset = dataset[:]

        # add bias input
        for i, row in enumerate(dataset):
            dataset[i] = ((1, ) + row[0], row[1])

        weights = [0.0 for i in dataset[0][0]]

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
                if not self.min_error or self.min_error and self.min_error['error_count'] > error_count:
                    self.min_error = {
                        'weights': weights,
                        'error_count': error_count,
                    }

            i += 1

            if i == max_loops:
                print 78 * '*'
                print 'Could not set weights without errors'
                self.weights = self.min_error['weights']
                return True
                # raise Exception('Could not set weights')
