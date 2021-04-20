from abc import ABC
from multiprocessing import Queue, Pipe, connection
from threading import Thread
from typing import Callable

from library.face_antspoofing import SpoofingDetector
from library.face_detector import FaceDetector

__all__ = ['Message', 'FaceDetectorWorker', 'SpoofingDetectorWorker']


class Message:
    __NO_DATA = b"0x00"

    def __init__(self, request_data, data_fetcher: connection.Connection = None):
        self.__respond_data = Message.__NO_DATA
        self.__data_fetcher = data_fetcher
        self.__request_data = request_data

    @property
    def request_data(self):
        return self.__request_data

    @property
    def respond_data(self):
        if self.__respond_data == Message.__NO_DATA:
            self.__respond_data = self.__data_fetcher.recv()
            self.__data_fetcher.close()
        return self.__respond_data


class Worker(Thread):
    __STOP_SIGNAL = b'U1RPUA=='

    def __init__(self, name=None):
        super().__init__(name=name)
        self.queue_request = Queue()
        self.task_count = 0
        self.__stop = 1

    def stop(self):
        self.queue_request.put_nowait(Worker.__STOP_SIGNAL)

    def __repr__(self):
        return self.__class__.__name__

    def processor(self) -> Callable:
        raise NotImplementedError

    def run(self):
        processor = self.processor()

        while self.__stop:
            data_pipe = self.queue_request.get()
            if data_pipe == Worker.__STOP_SIGNAL:
                break

            args, kwargs = data_pipe.recv()
            result = processor(*args, **kwargs)
            data_pipe.send(result)

    def request(self, *args, **kwargs) -> Message:
        request, respond = Pipe()
        msg = Message((args, kwargs), respond)
        self.queue_request.put(request)
        respond.send((args, kwargs))
        return msg

    __call__ = request


class FaceDetectorWorker(Worker, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__("FaceDetector")
        self.args = args
        self.kwargs = kwargs

    def processor(self):
        model = FaceDetector(*self.args, **self.kwargs)
        return model


class SpoofingDetectorWorker(Worker, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__("SpoofingDetector")
        self.args = args
        self.kwargs = kwargs

    def processor(self):
        spoofing_detector = SpoofingDetector(*self.args, **self.kwargs)
        return spoofing_detector
