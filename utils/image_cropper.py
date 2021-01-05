from PIL import Image
import math
import numpy as np
import diagonal_crop
import random
import os
import tqdm
import matplotlib.pyplot as plt
import cv2

def namefinder(path):
    dirlist = [int(i.split('.')[0]) for i in os.listdir(path)]
    if len(dirlist) is 0:
        return 0
    return np.max(np.asarray(dirlist))


class Cropper(object):
    def __init__(self, img_path, mask_path, target_size, delta, target_save, mask_save, origin=None, offset=None):
        self.img_path = img_path
        self.target_save = target_save
        self.mask_save = mask_save
        self.mask_path = mask_path
        self.target_size = target_size
        self.image = Image.open(self.img_path)
        self.mask = Image.open(self.mask_path)
        if self.image.size != self.mask.size:
            raise ValueError('input image are not of the same size')
        self.size = self.image.size
        self.gen = CoordinateGen(image_size=list(self.size),
                                 target_size=self.target_size,
                                 delta=delta,
                                 origin=origin,
                                 offset=offset)
        if not os.path.exists(img_path):
            raise FileExistsError('image path not exist')
        if not os.path.exists(self.target_save):
            print('target save path not exist, creating for you')
            os.mkdir(self.target_save)
        if not os.path.exists(self.mask_save):
            print('mask save path not exist, creating for you')
            os.mkdir(self.mask_save)

    def random_angle_crop(self, base):
        if not isinstance(base, tuple):
            raise TypeError('base should be a tuple with 2 elements')
        state = True
        counter = 0
        while state:
            angle = math.pi * random.random()
            height = self.target_size
            width = self.target_size
            cropped = diagonal_crop.crop(self.image, base, angle, height, width)
            cropped_mask = diagonal_crop.crop(self.mask, base, angle, height, width)
            counter += 1
            if Cropper.blank_check(cropped, 0.15):
                print(f'rotating for {angle * 180 / math.pi}')
                state = False
            elif counter >= 10:
                print('fail to rotate, keep the original image as output')
                cropped = self.image.crop((base[0], base[1],
                                           base[0] + self.target_size,
                                           base[1] + self.target_size))
                cropped_mask = self.mask.crop((base[0], base[1],
                                               base[0] + self.target_size,
                                               base[1] + self.target_size))
                state = False
        return cropped, cropped_mask

    def Crop(self):
        target_naming_start = namefinder(self.target_save)
        mask_naming_start = namefinder(self.mask_save)
        if target_naming_start != mask_naming_start:
            raise ValueError('expecting to have both save path having the same number of images')
        else:
            start = target_naming_start
        for i, coordinate in tqdm.tqdm(enumerate(self.gen)):
            if random.random() < 0.9:
                cropped, cropped_mask = self.random_angle_crop(base=(coordinate[0], coordinate[1]))
                cropped.save(os.path.join(self.target_save, str(start + 1 + i) + '.png'))
                cropped_mask.save(os.path.join(self.mask_save, str(start + 1 + i) + '.png'))
            else:
                cropped = self.image.crop(coordinate)
                cropped_mask = self.mask.crop(coordinate)
                cropped.save(os.path.join(self.target_save, str(start + 1 + i) + '.png'))
                cropped_mask.save(os.path.join(self.mask_save, str(start + 1 + i) + '.png'))

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
        self.image_size = image_size
        self.target_size = target_size
        self.origin[0] = offset[0] + self.origin[0]  # left bound
        self.origin[1] = offset[1] + self.origin[1]  # upper bound
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
        print(f'can generate {self.v_image_num * self.h_image_num} image from this big one')

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

    def __len__(self):
        return self.v_image_num * self.h_image_num


if __name__ == "__main__":
    Image.MAX_IMAGE_PIXELS = 2000000000  # make sure you have 16GB or 32GB memory...
    # crop = Cropper(img_path=r'C:\Users\Tim Wang\Desktop\large satellite images\wz\src\image_4.png',
    #                mask_path=r'C:\Users\Tim Wang\Desktop\large satellite images\wz\label\label_4.png',
    #                target_size=1024,
    #                delta=[512, 512],
    #                target_save=r'C:\Users\Tim Wang\Desktop\large satellite images\cropped_wz_src',
    #                mask_save=r'C:\Users\Tim Wang\Desktop\large satellite images\cropped_wz_mask')
    # crop.Crop()
    # print(namefinder(r'C:\Users\Tim Wang\Desktop\large satellite images\cropped_cz_src'))
    CoordinateGen(image_size=[16360, 7728])
    # img = cv2.imread(r'D:\SpaceNet\AOI_2_Vegas\PS-RGB\SN3_roads_train_AOI_2_Vegas_PS-RGB_img5.tif', cv2.IMREAD_UNCHANGED)
    # img = img/img.max()
    # img = img*255
    # img = img.astype(np.uint8)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # plt.imshow(img)
    # plt.show()