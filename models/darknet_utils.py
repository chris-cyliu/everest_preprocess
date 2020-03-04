import cv2
import gc
import numpy as np
from ctypes import *


__all__ = ['darknet_resize']


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

lib = CDLL("darknet/libdarknet.so", RTLD_GLOBAL)

resize_image = lib.resize_image
resize_image.argtypes = [IMAGE, c_int, c_int]
resize_image.restype = IMAGE

free_image = lib.free_image
free_image.argtypes = [IMAGE]


@profile
def array_to_image(arr):
    # share memory
    arr = arr.transpose(2, 0, 1)
    c, h, w = arr.shape[:3]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32)
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w, h, c, data)
    return im, arr


@profile
def image_to_array(im, shape):
    # share memory
    # python won't free c objects
    arr = np.ctypeslib.as_array(im.data, shape=(shape[2], shape[0], shape[1]))
    arr = arr.transpose((1, 2, 0))
    return arr


@profile
def darknet_resize(im, shape):
    # shape: (h, w)
    image, _ = array_to_image(im)
    image_resize = resize_image(image, shape[1], shape[0])
    image_resize_np = image_to_array(image_resize, shape)
    free_image(image_resize)
    return image_resize_np


@profile
def test_darknet_resize():
    image_path = 'darknet/data/dog.jpg'

    a = cv2.imread(image_path)
    ar = darknet_resize(a, (416, 416, 3))
    del a
    del ar
    gc.collect()

    b = cv2.imread(image_path)
    br = darknet_resize(b, (416, 416, 3))
    del b
    del br
    gc.collect()

    c = cv2.imread(image_path)
    cr = darknet_resize(c, (416, 416, 3))
    del c
    del cr
    gc.collect()

    """
    image_resize_cv2 = cv2.resize(image, (416, 416), interpolation=cv2.INTER_LINEAR)
    print(image_resize_cv2.shape)
    """


"""
python3 -m memory_profiler models/darknet_utils.py
"""
if __name__ == '__main__':
    test_darknet_resize()
