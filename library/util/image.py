import os

import cv2
import numpy

from library.util import check_type
from library.util.constant import FLIP_HORIZONTAL, FLIP_VERTICAL, FLIP_BOTH, DEFAULT_QUALITY

"""
This tool implement from OpenCV. Can you find more options at https://github.com/opencv/opencv
Drawing: https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html

* All method still cover (WIDTH, HEIGHT)
"""

IM_RGB = 0
IM_BGR = 0


def imread(img_path, pixel_format=IM_RGB):
    check_type("img_path", img_path, str)

    if not os.path.isfile(img_path):
        raise FileNotFoundError(img_path)

    image = cv2.imread(img_path)
    if pixel_format == IM_RGB:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def imwrite(img, img_path, quality=DEFAULT_QUALITY, pixel_format=IM_RGB, over_write=False):
    check_type("img_path", img_path, str)

    if os.path.isfile(img_path) and not over_write:
        raise FileExistsError(img_path)

    if pixel_format != IM_RGB:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.imwrite(f"{img_path}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])


def imdecode(buf, flag=cv2.IMREAD_COLOR, pix_fmt=IM_RGB):
    check_type("buf", buf, bytes)

    buf = numpy.frombuffer(buf, dtype='uint8')
    image = cv2.imdecode(buf, flag)
    if pix_fmt == IM_RGB:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def imencode(image, pix_fmt=IM_RGB, quality=DEFAULT_QUALITY):
    check_type("image", image, numpy.ndarray)

    if pix_fmt == IM_RGB:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    _, buf = cv2.imencode('.jpeg', image, params=[cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return buf


def resize(img, width=None, height=None, interpolation=None):
    """
    This function resize with keep ratio supported. Auto downscale or upscale fit with image's height.
    if width == -1 or height == -1 mean auto scale
    """
    assert isinstance(img, numpy.ndarray)

    # check any width or height parameters was filled.
    if (width is None and height is None) \
            or not ((not width or width > 0) or (not height or height > 0)) \
            or (width == img.shape[1] and height == img.shape[0]):
        return img

    old_h, old_w, _ = img.shape
    if not width or width <= 0:
        width = height / old_h * old_w

    if not height or height <= 0:
        height = width / old_w * old_h

    if interpolation is not None:
        return cv2.resize(img, (int(width), int(height)), interpolation=interpolation)
    return cv2.resize(img, (int(width), int(height)))


__all__ = ['imencode', 'imdecode', 'imread', 'imwrite', 'resize']
