import logging
from typing import Union
from urllib.parse import unquote

import click
import nptyping as npt
import requests
from fastapi import UploadFile, HTTPException, Form
from starlette.datastructures import UploadFile as ULFile

from library.util.image import imdecode


def get_logger(name: str = "API", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def image_read(image: Union[str, UploadFile] = Form(...)) -> npt.NDArray[npt.UInt8]:
    """
    Read image from request. Support binary image from Form and read image from URL.

    Parameters
    ----------
    image: request image. FileIO or URL

    Returns
    -------
    Image in numpy array
    """
    logger = get_logger()

    # check filled
    if image is None:
        raise HTTPException(400, "Require field `image`")

    # adapter with many type of image field
    if isinstance(image, str):
        image = unquote(image)
        try:
            rq = requests.get(image)
        except Exception:
            raise HTTPException(400, "Wrong URL format!") from None

        if rq.status_code != 200:
            raise HTTPException(400, "Image's URL isn't correct")
        buf = rq.content
        logger.info(f"REQUEST URL: {image}")
    elif isinstance(image, ULFile):
        if not image.content_type.startswith("image"):
            raise HTTPException(400, "File content must be 'image'")
        buf = image.file.read()
        click.echo(f"REQUEST URL: {image.filename}")
    else:
        raise HTTPException(400, "Params type not support!")

    # decode image
    if not len(buf):
        raise HTTPException(400, "File empty error!")

    try:
        image = imdecode(buf)
    except Exception:
        raise HTTPException(400, "Image format error!") from None
    return image
