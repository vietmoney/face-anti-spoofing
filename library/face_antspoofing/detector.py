from typing import Any, Sequence, Tuple

import cv2
import nptyping as npt
import numpy as np
import torch
from torch.nn.functional import softmax

from library.models.mini_fasnet import MiniFASNetV1SE, MiniFASNetV2
from library.util import scale_box, remove_prefix
from library.util.transform import Transform, resize

__all__ = ['MulFasNet', 'SpoofingDetector']
_CPU_DEVICE = "cpu"


class MulFasNet:
    def __init__(self, model_path: str, device=_CPU_DEVICE):
        model_load = torch.load(model_path, map_location=device)
        self.models = [MiniFASNetV1SE(conv6_kernel=(5, 5)), MiniFASNetV2(conv6_kernel=(5, 5))]

        # Load model
        for idx, model in enumerate(self.models):
            model.load_state_dict(remove_prefix(model_load[model.__class__.__name__], 'module.'), strict=False)
            model.to(device)
            model.eval()

        self.max_scale = [4, 2.7]
        self.device = device

    def forward(self, faces_scale: Sequence[torch.Tensor]) -> npt.NDArray[(Any, 3), npt.Float]:
        predict = torch.zeros(faces_scale[0].size(0), 3, dtype=torch.float64, device=self.device)
        with torch.no_grad():
            for model, images in zip(self.models, faces_scale):
                images = images.to(self.device)
                result = model(images)
                result = softmax(result, dim=1)
                predict += result

        if self.device != _CPU_DEVICE:
            predict = predict.detach().cpu()
        return predict.detach().numpy()

    __call__ = forward


class SpoofingDetector:
    def __init__(self, model_path: str, device=_CPU_DEVICE, face_size=80):
        self.model = MulFasNet(model_path, device=device)
        self.device = self.model.device
        self.face_size = face_size

        self.transform = Transform([
            lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2BGR),
            resize(face_size, face_size),
            lambda x: torch.from_numpy(x.transpose((2, 0, 1))).float()
        ])

    def predict(self, boxes: Sequence[Sequence[int]],
                image: npt.NDArray[npt.UInt8]) -> Sequence[Tuple[bool, float]]:
        faces_scale = list()
        for scale in self.model.max_scale:
            face_tensor = torch.zeros(len(boxes), 3, self.face_size, self.face_size, dtype=torch.float32)
            for idx, box in enumerate(boxes):
                height, width = image.shape[:2]
                box_scaled = scale_box(width, height, box, scale)
                face_img = image[box_scaled[1]:box_scaled[3], box_scaled[0]:box_scaled[2]]
                face_tensor[idx] = self.transform(face_img)
            faces_scale.append(face_tensor)

        predict = self.model(faces_scale)
        labels = np.argmax(predict, axis=1)
        return [(label == 1, np.round(predict[idx][label] / 2., 4)) for idx, label in enumerate(labels)]

    __call__ = predict
