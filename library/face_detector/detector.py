__author__ = "tindht@vietmoney.vn"
__copyright__ = "Copyright 2021, VietMoney Face Anti-spoofing"


from typing import List, Tuple

import nptyping as npt
import numpy
import torch

from library.models.retina_face import py_cpu_nms, decode, decode_landm, PriorBox, RetinaNet
from library.util.image import resize
from library.util.transform import Transform
from library.util.vector import euclidean_distance

__all__ = ['FaceDetector']
_CPU_DEVICE = "cpu"


class RetinaFace(object):
    """
    FaceDetector implement RetinaFace

    Parameters
    ----------
    model_path: str
        Path of pre-trained model

    device: str
        (Default: cpu) CPU or GPU ('cuda[:gpu_id]')
    """

    def __init__(self, model_path, device=_CPU_DEVICE):
        self.model = RetinaNet()
        self.prior_box = PriorBox(self.model.cfg)
        self.device = device

        self.image_size = None
        self.prior_data = None
        self.landmark_scale = None
        self.box_scale = None

        self.transform = Transform([
            lambda x: x.astype(numpy.float32) - (104, 117, 123),
            lambda x: torch.from_numpy(x.transpose((2, 0, 1))).float().unsqueeze(0),
            lambda x: x.to(self.device)
        ])

        pretrained_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(pretrained_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

    def update_prior(self, image_size: Tuple[int, ...]):
        if self.image_size == image_size:
            return

        priors = self.prior_box(image_size, self.device)
        self.prior_data = priors.data
        self.image_size = image_size
        self.landmark_scale = torch.tensor(
            [image_size[1], image_size[0], image_size[1], image_size[0],
             image_size[1], image_size[0], image_size[1], image_size[0],
             image_size[1], image_size[0]],
            dtype=torch.float32
        ).to(self.device)
        self.box_scale = torch.tensor([image_size[1], image_size[0], image_size[1], image_size[0]], dtype=torch.float32)
        self.box_scale = self.box_scale.to(self.device)

    def decode(self, locations: torch.Tensor,
               confident: torch.Tensor,
               landms: torch.Tensor,
               threshold: float,
               top=500) -> numpy.ndarray:
        """
        Decode and filter predict data from model

        Parameters:
            locations: list of face's location
            confident: list of face's score
            landms: list of face's land mark
            threshold: filter face's score threshold
            top: top face's score

        Returns:
            numpy.ndarray

            Detected face include: box, score, land_mark.
            - box.dtype: numpy.int32
            - score.dtype: numpy.float32
            - landmark.dtype: numpy.float32
        """
        boxes = decode(locations.data.squeeze(0), self.prior_data, self.model.cfg['variance'])
        boxes = boxes * self.box_scale
        scores = confident.squeeze(0).data[:, 1]

        landms = decode_landm(landms.data.squeeze(0), self.prior_data, self.model.cfg['variance'])
        landms = landms * self.landmark_scale

        inds = scores > threshold

        # ignore low scores
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # convert to cpu if device != cpu
        if self.device != _CPU_DEVICE:
            boxes = boxes.cpu()
            landms = landms.detach().cpu()
            scores = scores.detach().cpu()

        boxes = boxes.detach().numpy()
        landms = landms.detach().numpy()
        scores = scores.detach().numpy()

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = numpy.hstack((boxes, scores[:, numpy.newaxis])).astype(numpy.float32, copy=False)
        keep = py_cpu_nms(dets, 0.3)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:top, :]
        landms = landms[:top, :]

        faces = numpy.concatenate((dets, landms), axis=1)

        # remove box out of image's boundary
        faces = faces[[numpy.any((face[:4] >= 0) & (face[:4] < 5000)) for face in faces]]

        # sort by box size
        if len(faces) > 1:
            faces_size = numpy.array([euclidean_distance(face[:2], face[2:4]) for face in faces])
            faces_size = faces_size[faces_size > 24]
            order = numpy.argsort(faces_size)[::-1]
            faces = faces[order]
        return faces

    def detect(self, image: numpy.ndarray, threshold, top=500) \
            -> List[Tuple[npt.NDArray[npt.Float], numpy.float32, npt.NDArray[npt.Float]]]:
        """
        Detect faces in image.

        Parameters
        ----------
            image: image source
            threshold: face threshold
            top: top k of faces.

        Yields
        -------
            List of face include [box, score, land_mark]
        """
        # update decoder.
        self.update_prior(image.shape[:2])

        # transform image
        transformed_img = self.transform(image)

        # forward
        loc_encoded, score_encoded, landms_encoded = self.model(transformed_img)
        faces_decoded = self.decode(loc_encoded, score_encoded, landms_encoded, threshold, top)

        # numpy -> Face
        for face in faces_decoded:
            box = face[:4]
            score = face[4]
            land_mark = face[5:].reshape(-1, 2)
            yield box, score, land_mark

    __call__ = detect


class FaceDetector(object):
    """
    Face detector implement from RetinaFace: https://github.com/biubug6/Pytorch_Retinaface
    with scale step that speedup and normalize input data.
    """

    def __init__(self, model_path,
                 detect_threshold=0.975,
                 scale_size=480,
                 device='cpu'):
        """
        Parameters
        ----------
            model_path: Path of pre-trained model
            detect_threshold: Threshold of confidence score of detector
            scale_size: Scale size input image. `Recommend in [240, 1080]`
            device: device model loaded in. (Default: cpu)
        """
        # prepare face detector
        self.retina_face = RetinaFace(model_path, device=device)

        self.scale_size = scale_size
        if scale_size < 240:
            raise ValueError("Scale factor too small. scale_size >= 240 px")
        self.detect_threshold = detect_threshold

    def process(self, image) -> List[Tuple[List[int], float, List[List[int]]]]:
        """
        Post process of face detected from model

        Parameters
        ----------
            image: image source

        Returns
        -------
            List of face with raw resolution
        """
        # scaling source -> speed up face detect process
        height, width = image.shape[:2]
        scale = max(height / self.scale_size, 1.)
        image_scaled = image

        # resize to scale size
        if scale > 1.:
            image_scaled = resize(image, width=-1, height=self.scale_size)

        detected_faces = self.retina_face(image_scaled, self.detect_threshold)

        faces = list()
        for box, score, land_mark in detected_faces:
            # scale
            box = (box * scale).astype(numpy.int32)
            land_mark = (land_mark * scale).astype(numpy.int32)
            score = score.astype(numpy.float32)
            faces.append((box, score, land_mark))
        return faces

    __call__ = process
