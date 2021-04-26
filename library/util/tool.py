import cv2
import numpy

__all__ = ['show_image', 'destroy_windows', 'draw_square']


def show_image(img, windows_name, windows_size=(640, 360), windows_mode=cv2.WINDOW_NORMAL, wait_time=1,
               key_press_exit="q"):
    """
    Show image in RGB format

    Parameters
    ----------
    img: numpy.ndarray
        image array

    windows_name: str
        Title of window

    windows_size: tuple[int, int]
        (Default: SD_RESOLUTION) size of window

    windows_mode: int
        (Default: cv2.WINDOW_NORMAL) Mode of window

    wait_time: int
        Block time. (-1: infinite)

    key_press_exit: str
        Key stop event.

    Returns
    -------
    bool
        True - Stop event from user
    """
    cv2.namedWindow(windows_name, windows_mode)
    cv2.imshow(windows_name, img[:, :, ::-1])
    cv2.resizeWindow(windows_name, *windows_size)

    if cv2.waitKey(wait_time) & 0xFF == ord(key_press_exit):
        cv2.destroyWindow(windows_name)
        return False
    return True


def destroy_windows(*windows_name):
    """
    Destroy windows if set. Else destroy all

    Parameters
    ----------
    windows_name: str
        List windows name. Empty same mean all windows.
    """
    if windows_name:
        for window_name in windows_name:
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass
    else:
        cv2.destroyAllWindows()


def draw_square(img, position, color=(0, 255, 0)) -> numpy.ndarray:
    """
    Draw text at position in image.
    - position: top-left, bottom-right of square
    - color: support tuple and hex_color
    :return:
    """
    if not isinstance(position, tuple):
        position = tuple(position)

    return cv2.rectangle(img, position[0:2], position[2:4], color, 2)


def draw_text(img, label, position, color=(0, 0, 255), scale_factor=1, thickness=1,
              font=cv2.FONT_HERSHEY_DUPLEX, wrap_text=False) -> numpy.ndarray:
    """
    Draw text at position in image.
    - position: top-left of text
    - color: support tuple and hex_color
    :return:
    """
    if not isinstance(position, tuple):
        position = tuple(position)

    cv2.putText(img, label, position,
                fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale_factor,
                color=tuple(numpy.array(color)[::-1].tolist()), thickness=thickness)
    return img
