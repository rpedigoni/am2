am2
===

Stuff made on the machine learning course at my university


Simple neuron
-------------

Instantiating a neuron with a sigmoid transfer function: ::

    from am2 import Neuron, SigmoidTransferFunction

    neuron = Neuron(SigmoidTransferFunction, weights=[0, 1, 1])
    print neuron.run([3, 2, 1])


Or with a step transfer function: ::

    from am2 import Neuron, StepTransferFunction

    neuron = Neuron(StepTransferFunction, function=lambda y: y > 2, weights=[0, 1, 1])
    print neuron.run([1, 1, 1])


``function`` is optional (default is ``lambda y: y >= 1``)


Perceptron
----------

A simple Perceptron_ is implemented: ::

    from am2 import Perceptron

    train_dataset = [
        ((1, 0, 0), 1),
        ((1, 0, 1), 1),
        ((1, 1, 0), 1),


        ((1, 1, 1), 0),
    ]

    perceptron = Perceptron(function=lambda y: y >= 1)
    perceptron.train(train_dataset)  # executa o algoritmo de treinamento

    print perceptron.run((1, 0, 0))


.. _Perceptron: http://en.wikipedia.org/wiki/Perceptron


``matplotlib`` is used to plot graphs (you must have ``freetype`` and ``libpng`` installed)


