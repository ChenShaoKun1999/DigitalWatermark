from functools import partial
import numpy as np
from numpy.random import Generator, PCG64
from scipy.fftpack import dctn, idctn
from cv2 import cv2


def embed_watermark(arr:np.ndarray, seed:int, sigma:float, bit:bool) -> np.ndarray:
    '''returns IDCT(DCT(arr) + bit * Pseudo Noise), where pseudo noise obyes
    gaussian distribution
    '''
    freq = dctn(arr, norm='ortho')
    # Generate pesudo noise
    rg = Generator(PCG64(seed))
    pn = rg.normal(size=freq.shape)
    # Add pseudo noise to dct domain
    bit = 1 if bit else -1
    freq += bit * pn * sigma
    marked = idctn(freq, norm='ortho')
    return marked


def recover_watermark(arr:np.ndarray, seed:int) -> float:
    '''returns sum(DCT(arr) * Pseudo Noise)
    if return value is significantly larger than 0, watermark bit is 1
    if it's significantly smaller than 0, watermark bit is 0
    '''
    freq = dctn(arr, norm='ortho')
    # Generate pesudo noise
    rg = Generator(PCG64(seed))
    pn = rg.normal(size=freq.shape)
    # Recover the embedded bit
    return np.sum(freq * pn) / freq.size


def jpeg_compress(img:np.ndarray, quality:int=100):
    result, jpeg = cv2.imencode('.jpeg', img, (cv2.IMWRITE_JPEG_QUALITY, 100))
    if result:
        return cv2.imdecode(jpeg, cv2.IMREAD_UNCHANGED)
    else:
        raise ValueError('JPEG compress error')


def f2i(img, int_type=np.uint8):
    '''Converts float32 image (0 to 1) to int (for example, uint8, 0 to 255)
    Any value below 0 consiidered 0, and any value above 1 considered 1'''
    img[img > 1] = 1
    img[img < 0] = 0
    return np.asarray(img * np.iinfo(int_type).max, dtype=int_type)


def i2f(img):
    '''Converts int image (for example, uint8 image ranging from 0 to 255) to
    float (0 to 1)
    '''
    return np.asarray(img, dtype=np.float32) / np.iinfo(img.dtype).max


bgr2yuv = partial(cv2.cvtColor, code=cv2.COLOR_BGR2YUV)
yuv2bgr = partial(cv2.cvtColor, code=cv2.COLOR_YUV2BGR)


def show_img(img, winname='window'):
    while True:
        cv2.imshow(winname, img)
        key = cv2.waitKey()
        if key == 27:
            break


def embed_to_img(img, seed, watermark):
    '''watermark = (watermark intensity: float, bit: bool)
    '''
    f_yuv = bgr2yuv(i2f(img))
    f_yuv[:, :, 0] = embed_watermark(f_yuv[:, :, 0], seed, *watermark)
    return f2i(yuv2bgr(f_yuv))


def recover_from_img(img, seed):
    f_yuv = bgr2yuv(i2f(img))
    return recover_watermark(f_yuv[:, :, 0], seed)


def test():
    u_bgr = cv2.imread('../example images/dog-5723334_1920.jpg')
    seed = 73765346
    watermark = (0.0001, True)
    marked = embed_to_img(u_bgr, seed, watermark)
    show_img(marked)
    recovered = recover_from_img(marked, seed)
    print(recovered)
    return recovered


def main():
    test()
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())