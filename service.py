import signal
import sys
import time
import os
from typing import Optional, List, Union
from urllib.parse import unquote

import click
import requests
import uvicorn
import orjson
from fastapi import FastAPI, UploadFile, HTTPException, Form
from pydantic import BaseModel
from starlette.datastructures import UploadFile as ULFile

from library.task_manager import SpoofingDetectorWorker, FaceDetectorWorker
from library.util.image import imdecode, imread


class FaceDetectRespond(BaseModel):
    size: int
    boxes: List[List[int]]
    landmarks: List[List[List[int]]]
    images: Optional[List[str]] = None


class AntiSpoofingRespond(BaseModel):
    nums: int
    is_reals: Optional[List[bool]]
    scores: Optional[List[float]]


@click.group()
@click.option("--detector-model",
              default="data/pretrained/retina_face.pth.tar",
              help="Face detector model file path")
@click.option("--detector-threshold",
              default=0.95,
              type=float,
              help="Face detector model threshold")
@click.option("--detector-scale",
              default=720,
              type=int,
              help="Face detector model scale. >= 240")
@click.option("--spoofing-model",
              default="data/pretrained/fasnet_v1se_v2.pth.tar",
              help="Face anti-spoofing file path")
@click.option("--device",
              default="cpu",
              type=str,
              help="Device to load model.")
@click.version_option("1.0")
@click.pass_context
def main(ctx, detector_model, detector_threshold, detector_scale,
         spoofing_model, device):
    face_detector = FaceDetectorWorker(detector_model,
                                       detect_threshold=detector_threshold,
                                       scale_size=detector_scale,
                                       device=device)
    spoofing_detector = SpoofingDetectorWorker(spoofing_model, device)

    ctx.ensure_object(dict)
    ctx.obj['face_detector'] = face_detector
    ctx.obj['spoofing_detector'] = spoofing_detector


@main.command(help="Detect face in images")
@click.argument(
    "images",
    nargs=-1,
    required=True,
    type=click.Path(exists=True)
)
@click.option(
    "--json", "-j",
    help="Export result to json file",
    type=click.Path(exists=False, writable=True)
)
@click.option(
    "--quiet", "-q",
    help="Turn off STD output",
    is_flag=True
)
@click.option(
    "--count", "-c",
    help="Counting image during process",
    is_flag=True
)
@click.option(
    "--overwrite", "-y",
    help="Force write json file.",
    is_flag=True
)
@click.pass_context
def detect(ctx, images, json, quiet, count, overwrite):
    """CLI"""
    if not json and quiet:
        raise click.UsageError("No output selected!")

    results = list()
    face_detector = ctx.obj['face_detector']

    # start service
    face_detector.start()

    for idx, img_path in enumerate(images):
        image = imread(img_path)
        faces = face_detector(image).respond_data

        boxes = list()
        scores = list()
        landmarks = list()
        for box, score, lm in faces:
            boxes.append(box.tolist())
            scores.append(float(score))
            landmarks.append(lm.tolist())

        result = {
            "path": os.path.basename(img_path),
            "boxes": boxes,
            "scores": scores,
            "landmarks": landmarks
        }

        if not quiet:
            result_str = str(result)

            if count:
                result_str = f"{idx + 1}/{len(images)} >> " + result_str

            print(result_str)
        elif count:
            print(f"{idx + 1}/{len(images)}", end="\r")
        results.append(result)

    face_detector.stop()

    if not json:
        sys.exit(0)

    if os.path.isfile(json):
        if not overwrite:
            if not click.confirm(f"Overwrite `{os.path.basename(json)}`?", default=True):
                raise FileExistsError(json)

    with open(json, "wb") as f:
        f.write(orjson.dumps(results))


@main.command(help="Detect spoofing face in images")
@click.argument(
    "images",
    nargs=-1,
    required=True,
    type=click.Path(exists=True)
)
@click.option(
    "--json", "-j",
    help="Export result to json file",
    type=click.Path(exists=False, writable=True)
)
@click.option(
    "--quiet", "-q",
    help="Turn off STD output",
    is_flag=True
)
@click.option(
    "--count", "-c",
    help="Counting image during process",
    is_flag=True
)
@click.option(
    "--overwrite", "-y",
    help="Force write json file.",
    is_flag=True
)
@click.pass_context
def spoofing(ctx, images, json, quiet, count, overwrite):
    """CLI"""
    if not json and quiet:
        raise click.UsageError("No output selected!")

    results = list()
    face_detector = ctx.obj['face_detector']
    spoofing_detector = ctx.obj['spoofing_detector']

    # start service
    face_detector.start()
    spoofing_detector.start()

    for idx, img_path in enumerate(images):
        image = imread(img_path)
        faces = face_detector(image).respond_data

        if len(faces) <= 0:
            boxes = list()
            spoofs = list()
        else:
            boxes = [box.tolist() for box, _, _ in faces]
            spoofs = spoofing_detector(boxes, image).respond_data

        result = {
            "path": os.path.basename(img_path),
            "is_reals": [bool(is_spoof) for is_spoof, _ in spoofs],
            "scores": [float(score) for _, score in spoofs],
            "boxes": boxes
        }

        if not quiet:
            result_str = str(result)

            if count:
                result_str = f"{idx + 1}/{len(images)} >> " + result_str

            print(result_str)
        elif count:
            print(f"{idx + 1}/{len(images)}", end="\r")
        results.append(result)

    face_detector.stop()
    spoofing_detector.stop()

    if not json:
        sys.exit(0)

    if os.path.isfile(json):
        if not overwrite:
            if not click.confirm(f"Overwrite `{os.path.basename(json)}`?", default=True):
                raise FileExistsError(json)

    with open(json, "wb") as f:
        f.write(orjson.dumps(results))


@main.command(help="Run service as API")
@click.option("--host", default="127.0.0.1", type=str)
@click.option("--port", default=8000, type=int)
@click.option("--version", default="1.0.0", type=str)
@click.pass_context
def api(ctx, host, port, version):
    face_detector = ctx.obj['face_detector']
    spoofing_detector = ctx.obj['spoofing_detector']

    def signal_handler(*args, **kwargs):
        face_detector.stop()
        spoofing_detector.stop()

        while face_detector.is_alive and spoofing_detector.is_alive():
            time.sleep(0.25)

    # start service
    face_detector.start()
    spoofing_detector.start()

    _app = FastAPI(
        title="Vietmoney Face Attendance",
        version=version
    )

    @_app.get("/", description="Face Anti Spoofing - VietMoney")
    def hello():
        return "Welcome to Face Anti Spoofing - VietMoney\nAuthor: TinDang"

    @_app.post("/spoofing",
               response_model=AntiSpoofingRespond,
               summary="Detect spoofing face",
               description="Detect spoofing face from image and face's box")
    def anti_spoofing(image: Union[UploadFile, str] = Form(...)):
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
            click.echo(f"REQUEST URL: {image}")
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

        faces = face_detector(image).respond_data
        if len(faces) <= 0:
            raise HTTPException(400, "Can't detect any face in image.")

        boxes = [box.tolist() for box, _, _ in faces]
        spoof_msg = spoofing_detector(boxes, image)
        spoofs = spoof_msg.respond_data
        respond = {
            "nums": len(spoofs),
            "is_reals": [is_spoof for is_spoof, _ in spoofs],
            "scores": [score for _, score in spoofs],
            "boxes": boxes
        }
        click.echo(f"RESPOND: {respond}")
        return respond

    uvicorn.run(_app, host=host, port=port)
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame))
    print("Press Ctrl+C to stop.")
    signal.pause()


if __name__ == '__main__':
    main(auto_envvar_prefix='FA')
