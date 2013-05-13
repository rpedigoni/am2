# coding: utf-8
import unittest
from neuron import (Neuron, BaseTransferFunction,
                    SigmoidTransferFunction, StepTransferFunction)
from perceptron import Perceptron
from util import generate_classified_points


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


class StepTransferFunctionTest(unittest.TestCase):
    def test_step_true(self):
        neuron = Neuron(
            weights=[1, 2, 3],
            transfer_function=StepTransferFunction,
        )

        self.assertEqual(neuron.run([1, 2, 3]), 1)

    def test_step_false(self):
        neuron = Neuron(
            weights=[1, 2, 3],
            transfer_function=StepTransferFunction,
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

        perceptron = Perceptron(function=lambda x: x >= 1)
        perceptron.train(train_dataset)  # executa o algoritmo de treinamento

        v = perceptron.run((1, 0, 0))
        self.assertEqual(v, 1)

        v = perceptron.run((1, 1, 1))
        self.assertEqual(v, 0)

    def test_perceptron_2(self):
        train_dataset = [
            ((1, -1), False),
            ((2, -1), False),
            ((1, -3), False),
            ((4, -4), False),
            ((3, -2), False),
            ((5, -2), False),
            ((3, -4), False),
            ((-1, 2), True),
            ((-2, 3), True),
            ((-3, 2), True),
            ((-4, 4), True),
            ((-4, 1), True),
            ((-2, 4), True),
            ((-3, 3), True),
        ]

        perceptron = Perceptron(function=lambda x: x >= 0)
        perceptron.train(train_dataset)  # executa o algoritmo de treinamento
        perceptron.plot(train_dataset)
        raw_input('Enter to continue')

        v = perceptron.run((1, 1, 1))
        self.assertEqual(v, 0)

    def test_perceptron_generated_data(self):
        train_dataset = generate_classified_points(classes=[True, False], n=20)

        perceptron = Perceptron(function=lambda x: x >= 0)

        perceptron.train(train_dataset)
        perceptron.plot(train_dataset)
        print train_dataset
        raw_input('Enter to continue')

    def test_perceptron_generated_data_min_weights(self):
        train_dataset = generate_classified_points(classes=[True, False], n=20, distance=50)

        perceptron = Perceptron(function=lambda x: x >= 0)

        perceptron.train(train_dataset)
        perceptron.plot(train_dataset)
        perceptron.count_errors(train_dataset)

        raw_input('Enter to continue')

    def test_perceptron_generated_data_error(self):
        train_dataset = [((9.30194146152442, 54.29378711947825), True), ((2.1235211235782163, 42.41857119148967), True), ((1.359590385942031, 48.19605969472737), True), ((11.304173995362238, 40.21203508190859), True), ((15.491418600968164, 51.74125443774198), True), ((3.0448137332985663, 55.033225748928615), True), ((7.410534521463678, 48.82884207059357), True), ((7.965318834195054, 41.168243991915965), True), ((12.025772533772868, 44.82181684357318), True), ((-2.5480606577592253, 52.21817880722483), True), ((14.616754918016932, 27.56413924824511), False), ((13.735161526492831, 28.195520388962247), False), ((26.320312452059365, 37.52778678930581), False), ((28.50174788722075, 24.833317461626116), False), ((16.625494494802766, 35.423472182867286), False), ((19.135182106291616, 24.00082676846897), False), ((22.4174108797297, 36.127585975425156), False), ((12.439758335580695, 21.353479917856465), False), ((24.57194081489678, 32.46668179093647), False), ((28.556992040085298, 23.344536461376247), False)]
        perceptron = Perceptron(function=lambda x: x >= 0)

        perceptron.train(train_dataset)
        perceptron.plot(train_dataset)
        raw_input('Enter to continue')

if __name__ == '__main__':
    unittest.main()
