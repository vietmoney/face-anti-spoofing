from typing import Union, Iterator, Type

import numpy

__all__ = ['check_type', 'euclidean_distance', 'scale_box', 'remove_prefix']


def remove_prefix(state_dict, prefix):
    """Remove prefix of load_state_module"""
    if "state_dict" in state_dict.keys():
        state_dict = state_dict['state_dict']
    f = (lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x)
    return {f(key): value for key, value in state_dict.items()}


def check_type(param_name: str, value, _type: Union[Type, Iterator[Type]], none_acceptable=False):
    msg = f"Type of `{param_name}` must in `{_type}`. But got `{type(value)}`"

    if not none_acceptable and value is None:
        raise TypeError(msg)

    if value is not None:
        if not isinstance(_type, (list, tuple)):
            _type = _type,

        if not isinstance(value, tuple(_type)):
            raise TypeError(msg)
    return True


def euclidean_distance(x1, x2):
    x1 = numpy.asarray(x1)
    x2 = numpy.asarray(x2)

    assert type(x1) is numpy.ndarray or type(x2) is numpy.ndarray
    assert x1.shape == x2.shape
    assert len(x1.shape) <= 2

    if len(x1.shape) == 1:
        return numpy.sqrt(numpy.sum((x1 - x2) ** 2))

    return numpy.sqrt(numpy.sum((x1[:, numpy.newaxis, :] - x2[numpy.newaxis, :, :]) ** 2, axis=-1))


def scale_box(src_w, src_h, bbox, scale):
    x = bbox[0]
    y = bbox[1]
    box_w = bbox[2] - bbox[0]
    box_h = bbox[3] - bbox[1]

    scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))

    new_width = box_w * scale
    new_height = box_h * scale
    center_x, center_y = box_w / 2 + x, box_h / 2 + y

    left_top_x = center_x - new_width / 2
    left_top_y = center_y - new_height / 2
    right_bottom_x = center_x + new_width / 2
    right_bottom_y = center_y + new_height / 2

    if left_top_x < 0:
        right_bottom_x -= left_top_x
        left_top_x = 0

    if left_top_y < 0:
        right_bottom_y -= left_top_y
        left_top_y = 0

    if right_bottom_x > src_w - 1:
        left_top_x -= right_bottom_x - src_w + 1
        right_bottom_x = src_w - 1

    if right_bottom_y > src_h - 1:
        left_top_y -= right_bottom_y - src_h + 1
        right_bottom_y = src_h - 1

    return int(left_top_x), int(left_top_y), int(right_bottom_x), int(right_bottom_y)
