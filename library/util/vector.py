import numpy


def normalize_l2(x):
    assert isinstance(x, numpy.ndarray)
    return x / numpy.sqrt(numpy.sum((x ** 2), keepdims=True, axis=1))


def cosine_similarity(x1, x2, skip_normalize=False):
    if type(x1) is list:
        x1 = numpy.array(x1)

    if type(x2) is list:
        x2 = numpy.array(x2)

    assert type(x1) is numpy.ndarray or type(x2) is numpy.ndarray
    assert x1.shape[-1] == x2.shape[-1]
    assert len(x1.shape) == 2

    if not skip_normalize:
        x1 = normalize_l2(x1)
        x2 = normalize_l2(x2)
    return numpy.dot(x1, x2.T)


def euclidean_distance(x1, x2):
    x1 = numpy.asarray(x1)
    x2 = numpy.asarray(x2)

    assert type(x1) is numpy.ndarray or type(x2) is numpy.ndarray
    assert x1.shape == x2.shape
    assert len(x1.shape) <= 2

    if len(x1.shape) == 1:
        return numpy.sqrt(numpy.sum((x1 - x2) ** 2))

    return numpy.sqrt(numpy.sum((x1[:, numpy.newaxis, :] - x2[numpy.newaxis, :, :]) ** 2, axis=-1))


l2_distance = euclidean_distance
