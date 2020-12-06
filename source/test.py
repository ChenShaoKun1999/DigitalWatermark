import numpy as np
from numpy.fft import fft, fft2
from numpy.random import Generator, PCG64
from scipy.fftpack import dct, dctn, idctn
import matplotlib.pyplot as plt
from cv2 import cv2


def show_img(img: np.ndarray):
    while True:
        cv2.imshow('win', img)
        key = cv2.waitKey()
        if key == 27:
            break

def img_freq():
    # 大概看一下图像频谱长什么样子
    # img = cv2.imread('dog-5723334_1920.jpg')
    img = cv2.imread('pottery-5680464_1920.jpg')
    y = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:, :, 0]
    # f8 = np.sum([[fft2(y[i:i+8, j:j+8]) for i in range(0, y.shape[0], 8)] for j in range(0, y.shape[1], 8)])
    # f8 = fft2(y[0:8, 0:8])
    i = 678
    j = 678
    f8 = np.abs(dctn(y[i:i+8, j:j+8]))
    f8 = f8 / np.max(f8)
    print(f8)
    f8 = cv2.resize(f8, (256, 256), interpolation=cv2.INTER_NEAREST)
    while True:
        cv2.imshow('freq', f8)
        cv2.imshow('space', cv2.resize(y[i:i+8, j:j+8], (256, 256), interpolation=cv2.INTER_NEAREST))
        key = cv2.waitKey()
        if key == 27:
            break


def rand_test():
    # 测试随机序列的频谱

    # 一维随机序列
    # arr = np.random.normal(size=160)
    # f = np.abs(fft(arr))
    # fig, ax = plt.subplots(2)
    # del fig
    # ax[0].plot(arr)
    # ax[1].plot(f)
    # plt.show()

    # 二维随机序列
    # arr = np.random.normal(size=(16, 16))
    arr = np.array([[i + j for i in range(0, 8)] for j in range(0, 8)])
    f = dctn(arr)
    # jpeg有损压缩
    result, jpeg = cv2.imencode('.jpeg', arr, (cv2.IMWRITE_JPEG_QUALITY, 80))
    print(result)
    arr_jpeg = cv2.imdecode(jpeg, -1)
    f_jpeg = dctn(arr_jpeg)
    print(f - f_jpeg)
    while True:
        show = lambda name, arr, size=(400, 400): cv2.imshow(name, cv2.resize(arr / np.max(arr), size, interpolation=cv2.INTER_NEAREST))
        show('space', arr)
        show('frequency', f)
        show('compressed', arr_jpeg)
        show('f-compressed', f_jpeg)
        show('f-diff', np.abs(f - f_jpeg))
        key = cv2.waitKey()
        if key == 27:
            break


def encode_1bit(src_bgr: np.ndarray, bit: bool, seed: int) -> (np.ndarray, float):
    bit = 1 if bit else -1
    src_yuv = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2YUV)
    rg = Generator(PCG64(seed))
    pn = np.asarray(rg.normal(size=src_yuv[:, :, 0].shape), dtype=np.uint8)
    product = np.sum(src_yuv[:, :, 0] * rg.normal(size=src_yuv[:, :, 0].shape))  # 图像与Pseudo-Noise的内积
    src_yuv[:, :, 0] += np.asarray(bit * pn, np.uint8)
    return cv2.cvtColor(src_yuv, cv2.COLOR_YUV2BGR), product


def decode_1bit(stego_bgr: np.ndarray, seed: int)-> float:
    stego_yuv = cv2.cvtColor(stego_bgr, cv2.COLOR_BGR2YUV)
    rg = Generator(PCG64(seed))
    pn = rg.normal(size=stego_yuv[:, :, 0].shape)
    bit = np.sum(stego_yuv[:, :, 0] * pn)
    return bit


def test_1bit():
    img = cv2.imread('./example images/dog-5723334_1920.jpg')
    pixels = img.shape[0] * img.shape[1]
    seed = 3952947295
    stego, img_pn_product = encode_1bit(img, False, seed)
    recovered = decode_1bit(stego, seed)
    # show_img(img)
    # show_img(stego)
    print(f'{img_pn_product / pixels:.2f}, {recovered / pixels:.2f}')


def stego_dct(arr, seed, sigma, bit):
    freq = dctn(arr, norm='ortho')
    # Generate pesudo noise
    rg = Generator(PCG64(seed))
    pn = rg.normal(size=freq.shape)
    # Add pseudo noise to dct domain
    bit = 1 if bit else -1
    freq += bit * pn * sigma
    stego = idctn(freq, norm='ortho')
    return stego


def recover_dct(arr, seed):
    freq = dctn(arr, norm='ortho')
    # Generate pesudo noise
    rg = Generator(PCG64(seed))
    pn = rg.normal(size=freq.shape)
    # Recover the embedded bit
    return np.sum(freq * pn) / freq.size



def test_dct():
    bgr = cv2.imread('./example images/dog-5723334_1920.jpg')
    # Convert to float, prevent information lost
    bgr = np.asarray(bgr, dtype=np.float32) / np.iinfo(bgr.dtype).max
    seed = 73767346

    # 嵌入水印
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    sigma = 0.0001
    yuv[:, :, 0] = stego_dct(yuv[:, :, 0], seed, sigma, True)

    # 将图像恢复为RGB
    stego_bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    stego_bgr[stego_bgr > 1] = 1
    stego_bgr[stego_bgr < 0] = 0
    # 重新量化为255色
    stego_bgr = np.asarray(stego_bgr * 255, dtype=np.uint8)
    # jpeg压缩
    result, stego_jpeg = cv2.imencode('.jpeg', stego_bgr, (cv2.IMWRITE_JPEG_QUALITY, 100))
    del result
    stego_bgr = cv2.imdecode(stego_jpeg, cv2.IMREAD_UNCHANGED)
    # show_img(stego_bgr)

    # 恢复水印
    stego_bgr = np.asarray(stego_bgr, np.float32) / np.iinfo(stego_bgr.dtype).max
    stego_yuv = cv2.cvtColor(stego_bgr, cv2.COLOR_BGR2YUV)
    recovered = recover_dct(stego_yuv[:, :, 0], seed)
    print(recovered)


def main():
    # rand_test()
    # test_1bit()
    # cvt_test()
    test_dct()
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())