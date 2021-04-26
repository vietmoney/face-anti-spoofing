__author__ = "tindht@vietmoney.vn"
__copyright__ = "Copyright 2021, VietMoney Face Anti-spoofing"


import time
from abc import ABC
from multiprocessing import Queue, Pipe, connection
from threading import Thread
from typing import Callable, Any

from library.face_antspoofing import SpoofingDetector
from library.face_detector import FaceDetector

__all__ = ['Message', 'Worker', 'FaceDetectorWorker', 'SpoofingDetectorWorker', 'stop_worker']


class Message:
    """Respond data type of Worker"""
    __NO_DATA = b"0x00"

    def __init__(self, request_data: Any, data_fetcher: connection.Connection = None):
        """
        Parameters
        ----------
            request_data: input data to Worker
            data_fetcher: Connect of PIPE.
        """
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
    """Worker extend Thread for define data flow and process async"""
    __STOP_SIGNAL = b'U1RPUA=='

    def __init__(self, name=None):
        super().__init__(name=name or self.__class__.__name__)
        self.queue_request = Queue()
        self.task_count = 0
        self.__stop = 1

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

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


def stop_worker(*services: Worker):
    for service in services:
        if not isinstance(service, Thread):
            continue

        service.stop()
        time_estimate = 0
        while service.is_alive():
            time.sleep(0.1)
            time_estimate += 0.1
            if time_estimate >= 5:
                service.queue_request.close()
                break


class FaceDetectorWorker(Worker, ABC):
    """FaceDetectorWorker implement from Worker & FaceDetector"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def processor(self):
        model = FaceDetector(*self.args, **self.kwargs)
        return model


class SpoofingDetectorWorker(Worker, ABC):
    """SpoofingDetectorWorker implement from Worker & SpoofingDetector"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def processor(self):
        spoofing_detector = SpoofingDetector(*self.args, **self.kwargs)
        return spoofing_detector
