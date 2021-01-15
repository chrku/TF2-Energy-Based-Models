import numpy as np
import math


def fuse_images(width, height, images, img_width, img_height):
    really_big_image = None
    for i in range(width):
        big_image = None
        for j in range(height):
            cur_image = images[width * i + j].reshape(img_width, img_height)
            if big_image is not None:
                big_image = np.hstack((big_image, cur_image))
            else:
                big_image = cur_image
        if really_big_image is not None:
            really_big_image = np.vstack((really_big_image, big_image))
        else:
            really_big_image = big_image
    return really_big_image


# Reference: https://stackoverflow.com/questions/1174984/how-to-efficiently-calculate-a-running-standard-deviation
class RunningStats:
    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def standard_deviation(self):
        return math.sqrt(self.variance())
