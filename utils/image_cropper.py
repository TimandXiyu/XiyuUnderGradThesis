from PIL import Image
import math
import numpy as np
import diagonal_crop
import random
import os
import tqdm


class Cropper(object):
    def __init__(self, img_path, target_size, delta, save, origin=None, offset=None):
        self.img_path = img_path
        self.save = save
        self.target_size = target_size
        self.image = Image.open(self.img_path)
        self.size = self.image.size
        self.gen = CoordinateGen(image_size=list(self.size),
                                 target_size=self.target_size,
                                 delta=delta,
                                 origin=origin,
                                 offset=offset)

    def random_angle_crop(self, base):
        if not isinstance(base, tuple):
            raise TypeError('base should be a tuple with 2 elements')
        state = True
        while state:
                angle = math.pi * random.random()
                height = self.target_size
                width = self.target_size
                cropped = diagonal_crop.crop(self.image, base, angle, height, width)
                if Cropper.blank_check(cropped, 0.35):
                    print(f'rotating for {angle * 180 / math.pi}')
                    state = False
                else:
                    print(f'rotate too much, re-trying, {angle * 180 / math.pi}')
        return cropped

    def Crop(self):
        for i, coordinate in tqdm.tqdm(enumerate(self.gen)):
            if random.random() < 0.5:
                cropped = self.random_angle_crop(base=(coordinate[0], coordinate[1]))
                cropped.save(os.path.join(self.save, str(i) + '.png'))
            else:
                cropped = self.image.crop((coordinate))
                cropped.save(os.path.join(self.save, str(i) + '.png'))

    @staticmethod
    def blank_check(image, percent):
        image = np.asarray(image)
        if np.sum(image == [0, 0, 0]) / (image.shape[0] * image.shape[1] * image.shape[2]) >= percent:
            return False
        else:
            return True




class CoordinateGen(object):
    def __init__(self, image_size, target_size=1024, delta=None, origin=None, offset=None):
        if not isinstance(image_size, list):
            raise TypeError('image size should be a list with 2 elements')
        if delta is None:
            self.delta = [512, 512]
        else:
            self.delta = delta
        if origin is None:
            self.origin = [0, 0]
        else:
            self.origin = origin
        if offset is None:
            offset = [0, 0]
        self.delta = delta
        self.image_size = image_size
        self.target_size = target_size
        self.origin[0] = offset[0] + self.origin[0]  # left bound
        self.origin[1] = offset[1] + self.origin[1] # upper bound
        self.h_image_num = ((self.image_size[0] - self.origin[0]) // self.delta[
            0]) - 1  # calculate horizontal image count
        self.v_image_num = ((self.image_size[1] - self.origin[1]) // self.delta[
            1]) - 1  # calculate vertical image count
        if self.image_size[0] < (target_size + self.delta[0]) or self.image_size[1] < (target_size + self.delta[1]):
            raise ValueError('image not big enough to crop')
        self.h_anchor = []
        self.v_anchor = []
        for i in range(self.h_image_num):
            self.h_anchor.append(self.origin[0] + i * self.delta[0])
        for i in range(self.v_image_num):
            self.v_anchor.append(self.origin[1] + i * self.delta[1])
        self.count = 0

    def __next__(self):
        if self.count >= self.v_image_num * self.h_image_num:
            raise StopIteration
        r_index = self.count // self.h_image_num
        h_index = self.count % self.h_image_num
        self.count += 1
        return [self.h_anchor[h_index],
                self.v_anchor[r_index],
                self.h_anchor[h_index] + self.target_size,
                self.v_anchor[r_index] + self.target_size]

    def __iter__(self):
        return self


if __name__ == "__main__":
    crop = Cropper(r'C:\Users\Tim Wang\Desktop\large satellite images\wz\src\image_1.png',
                   target_size=1024,
                   delta=[512, 512],
                   save=r'C:\Users\Tim Wang\Desktop\large satellite images\cropped_wz_src')
    # print(crop.blank_check(Image.open(r'C:\Users\Tim Wang\Desktop\large satellite images\cropped_wz_src\0.png'), percent=0.3))
    crop.Crop()

