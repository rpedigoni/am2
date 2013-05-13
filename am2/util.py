# coding: utf-8
import random


def generate_near_points(initial, distance, n=10):
    """
    Generates ``n`` points around the ``initial`` point,
    with a max distance ``d``.

    Usage example: generate_near_points((0.2, 0.5), 1.5, 15)
    """
    points = set()
    dimensions = len(initial)

    for i in range(0, n):
        new_point = []
        for d in range(0, dimensions):
            new_point.append(
                random.uniform(
                    initial[d] - distance,
                    initial[d] + distance,
                )
            )

        points.add(tuple(new_point))
    return points


def generate_classified_points(classes=[True, False], dimensions=2, distance=10, n=10):
    classified_points = []
    initial_points = []

    assert n % len(classes) == 0, '"n" must be disible by {0}'.format(len(classes))

    for cls in classes:
        # precisa validar se as áreas não são conflitantes (distance)
        # dimensões não estão sendo usadas

        initial_points.append(
            (
                random.randint(-50, 50),
                random.randint(-50, 50)
            )
        )

        for point in generate_near_points(
            (
                initial_points[-1][0],
                initial_points[-1][1],
            ),
            distance=distance,
            n=n / 2,
        ):
            classified_points.append((point, cls))

    return classified_points


def plot_dataset(dataset, color_map={}):
    import matplotlib.pyplot as pyplot

    # assert len(dataset[0][0]) == 2, 'data must be two-dimensional'

    if not color_map:
        # find all classes and map a color
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        classes = set()
        for row in dataset:
            classes.add(row[1])

        for cls in classes:
            color_map.update({cls: colors.pop()})

    for row in dataset:
        pyplot.plot(row[0][0], row[0][1], '{0}o'.format(color_map[row[1]]))


def slope_from_points(point1, point2):
    return (point2[1] - point1[1]) / (point2[0] - point1[0])


def plot_line(dataset, weights):
    """
    w0*1 + w1*x1 + w2*x2 = 0
    y = ax + b

    w2*x2 = -w1x1 - w0

    x2 = (-w1 / w2)*x1 + (-w0/w2)*1
    --   ---------- --   --------
    y  =      a      x       b
    """
    import matplotlib.pyplot as pyplot

    # assert len(dataset[0][0]) == 3, 'data must be two-dimensional'

    w0 = weights[0]
    w1 = weights[1]
    w2 = weights[2]

    # r = []
    _x = []
    _y = []

    for row in sorted(dataset, key=lambda x: x[0][0]):
        x1 = row[0][0]
        # x2 = row[0][1]
        c = ((-1 * w1) / w2) * x1 + ((-1 * w0) / w2) * 1
        _x.append(x1)
        _y.append(c)

    pyplot.plot(
        _x, _y,
    )


def count_errors(neuron, dataset):
    errors = {}
    for inputs, expected in dataset:
        result = neuron.run(inputs)

        if not result == expected:
            if errors.get(expected):
                errors[expected] += 1
            else:
                errors[expected] = 1

    if errors:
        print '*' * 78
        print 'Errors found!'
        for cls, count in errors.items():
            print '- {0}: {1} errors'.format(cls, count)
