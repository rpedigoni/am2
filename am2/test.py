# coding: utf-8
import unittest
from neuron import (Neuron, BaseTransferFunction,
                    SigmoidTransferFunction, StaircaseTransferFunction)
from perceptron import Perceptron


class NeuronTest(unittest.TestCase):
    def test_integrator(self):
        neuron = Neuron(
            weights=[1, 2, 3],
            transfer_function=BaseTransferFunction,  # does nothing
        )

        self.assertEqual(neuron.integrator([1, 1, 1]), 6)


class SigmoidTransferFunctionTest(unittest.TestCase):
    def test_sigmoid(self):
        neuron = Neuron(
            weights=[1, 2, 3],
            transfer_function=SigmoidTransferFunction,
        )

        v = neuron.run([0, 0, 0])
        self.assertEqual(v, 0.5)


class StairCaseTransferFunctionTest(unittest.TestCase):
    def test_staircase_true(self):
        neuron = Neuron(
            weights=[1, 2, 3],
            transfer_function=StaircaseTransferFunction,
        )

        self.assertEqual(neuron.run([1, 2, 3]), 1)

    def test_staircase_false(self):
        neuron = Neuron(
            weights=[1, 2, 3],
            transfer_function=StaircaseTransferFunction,
            function=lambda p: p >= 7,  # any function can be used here
        )
        self.assertEqual(neuron.run([1, 1, 1]), 0)


class PerceptronTest(unittest.TestCase):
    def test_perceptron(self):
        train_dataset = [
            ((1, 0, 0), 1),
            ((1, 0, 1), 1),
            ((1, 1, 0), 1),
            ((1, 1, 1), 0),
        ]

        perceptron = Perceptron(function=lambda y: y >= 1)
        perceptron.train(train_dataset)  # executa o algoritmo de treinamento

        v = perceptron.run((1, 0, 0))
        self.assertEqual(v, 1)

        v = perceptron.run((1, 1, 1))
        self.assertEqual(v, 0)


if __name__ == '__main__':
    unittest.main()
