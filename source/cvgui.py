import os
from functools import partial
from collections import namedtuple
from tkinter import filedialog
from cv2 import cv2
import numpy as np

class Area(tuple):
    AreaTuple = namedtuple('AreaTuple', ('height', 'width'))

    @classmethod
    def new(cls, window, top_left_pos, area_shape):
        '''
        初始化。由于tuple的初始化比较魔法，尝试了init和new无果之后被迫用这个奇怪的方法

        Parameters:
            window: CvWindow
            top_left_pos: Array[2]，显示区域左上角的位置
            area_shape: Array[2]，显示区域的高度和宽度
        '''
        verticle = slice(top_left_pos[0], top_left_pos[0] + area_shape[0])
        horizontal = slice(top_left_pos[1], top_left_pos[1] + area_shape[1])
        obj = (verticle, horizontal)
        self = cls(obj)
        self.window = window
        self.window_shape = cls.AreaTuple(window.win_h, window.win_w)
        self.img_shape = cls.AreaTuple(*window.img.shape[0:2])
        return self

    @property
    def height(self):
        return self[0].stop - self[0].start

    @property
    def width(self):
        return self[1].stop - self[1].start

    def move(self, direction, distance=0.2):
        '''
        返回移动后的显示区域（self保持不变）

        parameters:
            direction: "up", "left"，需要向上/向左的时候就把距离改成负数
            distance: number, 移动距离与显示区域宽度（横向移动时）或者高度（纵向移动
                时）的比值。如果超出图片范围则停止移动

        Returns:
            moved Area object
        '''
        i = self[0].start
        j = self[1].start

        if direction == 'up':
            i -= int(distance * self.height)
            if i < 0:
                i = 0
            elif i + self.height > self.img_shape.height:
                i = self.img_shape.height - self.height
        elif direction == 'left':
            j -= int(distance * self.width)
            if j < 0:
                j = 0
            elif j + self.width > self.img_shape.width:
                j = self.img_shape.width - self.width
        else:
            raise ValueError(f'Area.move: Unknown direction {direction}')
        return Area.new(self.window, [i, j], [self.height, self.width])

    def scale(self, center, rate):
        '''
        缩放显示区域

        parameters:
            center: Array[int, int]，缩放不动点（但是当缩放超出边界时还是会移动）
            rate: 缩放比例

        Returns:
            scaled Area object
        '''
        # 计算缩放之后的高度、宽度。放大之后显示范围变窄，显示区宽高变小，所以用除法，
        height = int(self.height / rate)
        width = int(self.width / rate)

        # 由于int是向下取整的，放大到非常大（宽小于width / rate < width + 1，或者
        # 高小于相应的值时）就没办法通过比例缩小回来了。强行将width加一
        if rate < 1 and width < 1 / (1 - rate):
            width += 1

        # 如果没有充满窗口，就把宽度/高度扩张到窗口宽高比
        if height / width > self.window_shape.height / self.window_shape.height:
            width = int(self.window_shape.height * height / self.window_shape.width)
        else:
            height = int(self.window_shape.width * width / self.window_shape.height)

        # 控制不会超过图片宽高
        # 可能导致超范围的原因：缩小、放大之后扩张到窗口大小
        height = min(height, self.img_shape.height)
        width = min(width, self.img_shape.width)

        # 原来鼠标位置相对于屏幕位置。注意由于xy坐标和ij坐标是反过来的，center[0] = x对应j
        # 设鼠标坐标是(x, y)，缩放前后显示区域左边缘为left和left'，宽度width和width'
        # (x - left) / width = (x - left') / width'
        # left' = x - (x - left) * width' / width
        x, y = center
        left = int(x - (x - self[1].start) * width / self.width)
        top = int(y - (y - self[0].start) * height / self.height)

        # 调整top和left到图像范围。重写一个太麻烦了于是直接借助move里面的
        result = Area.new(self.window, (top, left), (height, width))
        result = result.move('up', 0)
        result = result.move('left', 0)

        return result


class CvWindow(object):
    def __init__(self, img):
        self.img = img
        if self.img is None:
            raise ValueError('Empty image')
        self.win_h, self.win_w = 600, 600       # window size
        self.display_area = Area.new(self, [0, 0], self.img.shape[0:2])
        self.render()

    def render(self):
        # 拉伸图片到窗口宽度，同时保持宽高比
        # 注意：处理之前的高、宽对应原图像素，if-else语句之后对应屏幕像素
        height, width = self.display_area.height, self.display_area.width
        if height / width > self.win_h / self.win_w:
            height, width = self.win_h, int(self.win_h * width / height)
        else:
            height, width = int(self.win_w * height / width), self.win_w

        # 显示区域
        display = np.array(self.img)
        scaled = cv2.resize(display[self.display_area], (width, height), interpolation=cv2.INTER_NEAREST)

        # 用边框将图片充满窗口
        top = (self.win_h - height) // 2
        bot = self.win_h - height - top
        left = (self.win_w - width) // 2
        right = self.win_w - width - left
        self.display = cv2.copyMakeBorder(scaled, top, bot, left, right, cv2.BORDER_CONSTANT, (0, 0, 0))
        # 为了将鼠标点击投影回原图位置的变量
        self.scaled_size = Area.AreaTuple(height, width)
        self.top_border = top
        self.left_border = left

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL:
            direction = np.sign(flags)    # 向上滚的时候flags会是一个很大的正数，因此用符号判断滚动方向
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                # 缩放
                self.display_area = self.display_area.scale(self.project(x, y), 1.1 ** direction)
            elif flags & cv2.EVENT_FLAG_SHIFTKEY:
                # 左右移动
                self.display_area = self.display_area.move('left', direction * 0.2)
            else:
                # 上下移动
                self.display_area = self.display_area.move('up', direction * 0.2)

    def on_trackbar(self, bar_name, value):
        self.__setattr__(bar_name, value)

    def show(self):
        # 显示窗口
        win_name = 'CvGui'
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win_name, self.on_mouse)
        key = None
        while True:
            cv2.imshow(win_name, self.display)
            key = cv2.waitKey(50)
            if key == 27:
                # ESC，退出
                break
            self.render()
        cv2.destroyWindow(win_name)


    def project(self, x, y):
        '''将窗口上x, y位置投影回原图'''
        x_ratio = (x - self.left_border) / self.scaled_size.width
        y_ratio = (y - self.top_border) / self.scaled_size.height

        x_abs = self.display_area[1].start + x_ratio * self.display_area.width
        y_abs = self.display_area[0].start + y_ratio * self.display_area.height

        return int(x_abs), int(y_abs)


def main():
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())