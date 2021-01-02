from functools import partial
import numpy as np
from numpy.random import Generator, PCG64
from scipy.fft import dctn, idctn
from cv2 import cv2

from cvgui import CvWindow

def embed_watermark(arr, seed:int, data:bytes, a=[-1, 0, 1], p=[0.1, 0.8, 0.1]):
    rg = Generator(PCG64(seed))
    for byte in data:
        for bit in [(byte >> i) & 1 for i in range(0, 8)]:
            pn = rg.choice(a, p=p, size=arr.shape)
            arr = arr + pn if bit else arr - pn
    # print(f'{np.sum(arr > 255)} pixels over 255, biggest{np.max(arr)}')
    # print(f'{np.sum(arr < 0)} pixels below 0, smallest {np.min(arr)}')
    arr[arr > 255] = 255
    arr[arr < 0] = 0
    return arr


def recover_watermark(arr, seed:int, length:int, a=[-1, 0, 1], p=[0.1, 0.8, 0.1]):
    rg = Generator(PCG64(seed))
    data = []
    for byte in range(0, length):
        byte = 0
        for i in range(0, 8):
            pn = rg.choice(a, p=p, size=arr.shape)
            byte += 1 << i if np.sign(np.sum(arr * pn)) > 0 else 0
        data.append(byte)
    return bytes(data)


bgr2yuv = partial(cv2.cvtColor, code=cv2.COLOR_BGR2YUV)
yuv2bgr = partial(cv2.cvtColor, code=cv2.COLOR_YUV2BGR)


def main():
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())