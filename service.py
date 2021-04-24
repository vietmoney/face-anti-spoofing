__author__ = "tindht@vietmoney.vn"
__copyright__ = "Copyright 2021, VietMoney Face Anti-spoofing"


import os
import signal
from typing import Sequence

import click
import numpy
import orjson
import uvicorn
from fastapi import FastAPI, HTTPException, Depends

from library.task_manager import stop_worker, SpoofingDetectorWorker, FaceDetectorWorker
from library.util.api import image_read, get_logger
from library.util.image import imread

FACE_DETECTOR_KEY = "fa_face_detector"
FACE_ANTI_SPOOFING = "fa_face_anti_spoofing"
ENVVAR_PREFIX = 'FA'


def write_json(data, json_path: str, overwrite=False):
    if os.path.isfile(json_path):
        if not overwrite:
            if not click.confirm(f"Overwrite `{os.path.basename(json_path)}`?", default=True):
                raise FileExistsError(json_path)

    with open(json_path, "wb") as f:
        f.write(orjson.dumps(data))


@click.group()
@click.option("--detector-model",
              default="data/pretrained/retina_face.pth.tar",
              help="Face detector model file path")
@click.option("--detector-threshold",
              default=0.95,
              type=float,
              help="Face detector model threshold.")
@click.option("--detector-scale",
              default=720,
              type=int,
              help="Face detector model scale. (>= 240px)")
@click.option("--spoofing-model",
              default="data/pretrained/fasnet_v1se_v2.pth.tar",
              help="Face anti-spoofing file path")
@click.option("--device",
              default="cpu",
              type=str,
              help="Device to load model.")
@click.version_option("1.0")
@click.pass_context
def main(ctx, detector_model: str, detector_threshold: float, detector_scale: int, spoofing_model: str, device: str):
    device = device.lower()
    face_detector = FaceDetectorWorker(detector_model,
                                       detect_threshold=detector_threshold,
                                       scale_size=detector_scale,
                                       device=device)
    spoofing_detector = SpoofingDetectorWorker(spoofing_model, device)

    ctx.ensure_object(dict)
    ctx.obj[FACE_DETECTOR_KEY] = face_detector
    ctx.obj[FACE_ANTI_SPOOFING] = spoofing_detector


@main.command(help="Detect face in images")
@click.argument(
    "images",
    nargs=-1,
    required=True,
    type=click.Path(exists=True)
)
@click.option(
    "--json", "-j",
    help="Export results to json file",
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
def detect(ctx, images: Sequence[str], json: str, quiet: bool, count: bool, overwrite: bool):
    """CLI"""
    if not json and quiet:
        raise click.UsageError("No output selected!")

    results = list()

    # start service
    with ctx.obj[FACE_DETECTOR_KEY] as face_detector:
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

            std_out = ""
            if count:
                std_out = f"{idx + 1}/{len(images)}\t"

            if not quiet:
                print(f"{std_out}{result}", flush=True)

            results.append(result)

    if json:
        write_json(results, json, overwrite)


@main.command(help="Detect spoofing face in images")
@click.argument(
    "images",
    nargs=-1,
    required=True,
    type=click.Path(exists=True)
)
@click.option(
    "--json", "-j",
    help="Export results to json file",
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
def spoofing(ctx, images: Sequence[str], json: str, quiet: bool, count: bool, overwrite: bool):
    """CLI"""
    if not json and quiet:
        raise click.UsageError("No output selected!")

    results = list()

    # start service
    with ctx.obj[FACE_DETECTOR_KEY] as face_detector,\
            ctx.obj[FACE_ANTI_SPOOFING] as spoofing_detector:
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

            std_out = ""
            if count:
                std_out = f"{idx + 1}/{len(images)}\t"

            if not quiet:
                print(f"{std_out}{result}", flush=True)
            results.append(result)

    if json:
        write_json(results, json, overwrite)


@main.command(help="Run service as API")
@click.option("--host", help="API host.", default="localhost", type=str)
@click.option("--port", help="API port.", default=8000, type=int)
@click.option("--version", help="API version.", default="1.0.0", type=str)
@click.pass_context
def api(ctx, host: str, port: int, version: str):
    face_detector = ctx.obj[FACE_DETECTOR_KEY]
    spoofing_detector = ctx.obj[FACE_ANTI_SPOOFING]

    # start service
    face_detector.start()
    spoofing_detector.start()

    app = FastAPI(
        title="Vietmoney Face Attendance",
        version=version
    )

    logger = get_logger()

    @app.get("/", description="Face Anti Spoofing - VietMoney")
    def hello():
        return "Welcome to Face Detect & Anti Spoofing - Viet Money\nAuthor: TinDang"

    @app.post("/spoofing",
              summary="Detect spoofing face",
              description="Detect spoofing face from image and face's box")
    def anti_spoofing(image: numpy.ndarray = Depends(image_read)):
        faces = face_detector(image).respond_data
        if len(faces) <= 0:
            raise HTTPException(400, "Can't detect any face in image.")

        boxes = [box.tolist() for box, _, _ in faces]
        spoof_msg = spoofing_detector(boxes, image)
        spoofs = spoof_msg.respond_data
        respond = {
            "nums": len(spoofs),
            "is_reals": [bool(is_spoof) for is_spoof, _ in spoofs],
            "scores": [round(float(score), 4) for _, score in spoofs],
            "boxes": boxes
        }
        logger.info(f"RESPOND: {respond}")
        return respond

    @app.post("/detect",
              summary="Detect face",
              description="Detect face from image and face's info")
    def face_detect(image: numpy.ndarray = Depends(image_read)):
        faces = face_detector(image).respond_data

        if len(faces) <= 0:
            raise HTTPException(400, "Can't detect any face in image.")

        boxes = list()
        scores = list()
        landmarks = list()
        for box, score, lm in faces:
            boxes.append(box.tolist())
            scores.append(round(float(score), 4))
            landmarks.append(lm.tolist())

        respond = {
            "nums": len(faces),
            "boxes": boxes,
            "scores": scores,
            "landmarks": landmarks
        }
        logger.info(f"RESPOND: {respond}")
        return respond

    uvicorn.run(app, host=host, port=port)
    signal.signal(signal.SIGINT, lambda sig, frame: stop_worker(face_detector, spoofing_detector))
    print("API have been down! Press Ctrl+C to exit.")


if __name__ == '__main__':
    main(auto_envvar_prefix=ENVVAR_PREFIX, show_default=True, help_option_names=['--help', '-h'], max_content_width=120)
