from datetime import datetime
import numpy as np
from numpy.random import Generator, PCG64
import matplotlib.pyplot as plt
from cv2 import cv2

import watermark as wm
from cvgui import CvWindow

def show_img(img, winname='window'):
    while True:
        cv2.imshow(winname, img)
        key = cv2.waitKey()
        if key == 27:
            break


def jpeg_compress(img:np.ndarray, quality:int=100):
    result, jpeg = cv2.imencode('.jpeg', img, (cv2.IMWRITE_JPEG_QUALITY, 100))
    if result:
        return cv2.imdecode(jpeg, cv2.IMREAD_UNCHANGED)
    else:
        raise ValueError('JPEG compress error')


def bit_err_count(b1:bytes, b2:bytes):
    i = 0
    for byte1, byte2 in zip(b1, b2):
        diff = byte1 ^ byte2
        while diff:
            i += diff & 1
            diff >>= 1
    return i


def spatial_capacity():
    # bgr = cv2.imread('../example images/dog-5723334_1920.jpg')
    bgr = cv2.imread('../example images/computer-5736011_1920.jpg')
    seed = np.random.randint(0, np.iinfo(np.int32).max)
    length = 10
    data = np.random.bytes(length)
    a = [-1, 0, 1]
    p = [0.01, 0.98, 0.01]

    # embed & recover data
    begin = datetime.now()
    marked = wm.embed_watermark(bgr, seed, data, a, p)
    mark_end = datetime.now()
    recovered = wm.recover_watermark(marked, seed, length, a, p)
    finish = datetime.now()

    # calculate bit error rate
    print(f'time spent on marking: {mark_end - begin}')
    print(f'time spent on recovering: {finish - mark_end}')
    # print(f'data: {data}, recovered:{recovered}')
    print(f'{bit_err_count(data, recovered)} / {8 * length} bits wrong')

    # estimate data capacity
    diff = marked - bgr
    SNR = np.sum(diff ** 2) / np.sum(bgr ** 2)
    C = np.log2(1 + SNR)
    print('ideal capacity:', C)
    print('experimental capacity', length / marked.size)

    # Show watermarked image
    CvWindow(np.asarray(marked, dtype=np.uint8)).show()
    # show_img(np.asarray(marked, dtype=np.uint8))


def main():
    spatial_capacity()
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())