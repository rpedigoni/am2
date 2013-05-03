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
        ):
            classified_points.append((point, cls))

    return classified_points
