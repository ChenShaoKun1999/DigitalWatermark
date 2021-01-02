from datetime import datetime
import numpy as np
from numpy.random import Generator, PCG64
import matplotlib.pyplot as plt
from cv2 import cv2

import watermark as wm

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


def embed_spatial(arr, seed:int, bit:bool):
    rg = Generator(PCG64(seed))
    pn = np.asarray(rg.choice([-1, 0, 1], p=[0.1, 0.8, 0.1], size=arr.shape), dtype=np.int16)
    pn = pn if bit else -pn
    result = np.asarray(arr + pn, np.uint8)
    result[arr + pn > 255] = 255
    result[arr + pn < 0] = 0
    return result


def recover_spatial(arr, seed:int):
    rg = Generator(PCG64(seed))
    pn = np.asarray(rg.choice([-1, 0, 1], p=[0.1, 0.8, 0.1], size=arr.shape), dtype=np.int16)
    return np.sum(arr * pn)


def test_spatial():
    bgr = cv2.imread('../example images/dog-5723334_1920.jpg')
    # bgr = np.ones_like(bgr, dtype=np.uint8) * 128
    seed = 73765346

    # 测试插入1和0的情况
    mark_1 = embed_spatial(bgr[:, :, 0], seed, True)
    mark_0 = embed_spatial(bgr[:, :, 0], seed, False)
    recovered_1 = recover_spatial(mark_1, seed)
    recovered_0 = recover_spatial(mark_0, seed)
    print(recovered_1, recovered_0)

    # wm.CvWindow(bgr).show()
    wm.CvWindow(mark_0).show()


def spatial_capacity():
    # bgr = cv2.imread('../example images/dog-5723334_1920.jpg')
    bgr = cv2.imread('../example images/computer-5736011_1920.jpg')
    seed = np.random.randint(0, np.iinfo(np.int32).max)
    length = 100
    data = np.random.bytes(length)
    p1 = 0.02
    p = [p1, 1 - 2*p1, p1]

    # embed & recover data
    begin = datetime.now()
    marked = wm.embed_watermark(bgr, seed, data, p=p)
    mark_end = datetime.now()
    recovered = wm.recover_watermark(marked, seed, length, p=p)
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
    # wm.CvWindow(np.asarray(marked, dtype=np.uint8)).show()


def main():
    # test_spatial()
    spatial_capacity()
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())