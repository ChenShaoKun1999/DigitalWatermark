import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2

import watermark as wm

def bytes_to_bits(b):
    bits = []
    binary = np.array([1 << i for i in range(0, 8)])
    for byte in b:
        bits.extend(map(bool, byte & binary))
    return bits


def capacity():
    bgr = cv2.imread('../example images/dog-5723334_1920.jpg')
    seed = 73765346
    sigma = 0.0001
    # bits = bytes_to_bits(b'a')
    bits = bytes_to_bits(bytes(np.random.randint(0, 255, (16, ))))
    watermark = (sigma, bits)
    marked = wm.embed_to_img(bgr, seed, watermark)
    # wm.show_img(marked)

    # Calculate ideal channel capacity
    # C = W * log2(1 + SNR). In our case, bandwidth W = 1 cycle/pixel
    diff = bgr - marked
    SNR = np.sum(diff ** 2) / np.sum(bgr ** 2)
    C = np.log2(1 + SNR)
    print('ideal capacity:', C)
    print('experimental capacity', len(watermark[1]) / bgr.size)

    # calculate bit error rate
    recovered = wm.recover_from_img(marked, seed, len(bits))
    recovered_bits = list(map(lambda x: True if x >= 0 else False, recovered))
    error_bits = len([i for i in range(0, len(bits)) if bits[i] != recovered_bits[i]])
    print(f'Bit error rate: {error_bits / len(bits)} ({error_bits} / {len(bits)})')
    # print(watermark[1])
    # print(recovered)

    # TODO
    # 恢复结果直方图


def main():
    capacity()
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())