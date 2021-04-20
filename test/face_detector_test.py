import unittest
from library.util.image import imread
from library.face_detector import FaceDetector


class TestFaceDetect(unittest.TestCase):
    def read_jpeg(self):
        self.assertIsNotNone(imread("images/face_test-001.jpg"), "asdasd")

    def detect_face(self):
        face_detector = FaceDetector("data/pretrained/retina_face.pth.tar",
                                     scale_size=720, device="cuda:0")

        image = imread("images/face_test-001.jpg")
        faces = face_detector(image)
        self.assertIsNotNone(faces)


if __name__ == '__main__':
    unittest.main()
