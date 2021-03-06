from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import logging
from PIL import Image
import utils.TTA_tools as TTA_tools


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix='', aug=True):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.aug = aug
        # assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        ids = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith('.')]
        ids = [int(x) for x in ids]
        ids = sorted(ids)
        self.ids = [str(x) for x in ids]


        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        # w, h = pil_img.size
        newW, newH = scale[0], scale[1]
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newW))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        if img_nd.shape[2] is 3:
            img_nd = TTA_tools.Normalize()(img_nd)
        else:
            img_nd = img_nd / 255
        img_trans = img_nd.transpose((2, 0, 1))
        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0]).convert('L')
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        if self.aug is True:
            sample = {'image': img, 'label': mask}
            sample = TTA_tools.RandomHorizontalFlip()(sample)
            # sample = TTA_tools.RandomRotate(degree=10)(sample)
            sample = TTA_tools.RandomVerticalFlip()(sample)
            sample = TTA_tools.RandomGaussianBlur()(sample)
            img = sample['image']
            mask = sample['label']
            img = TTA_tools.randomHueSaturationValue(img)

        img = self.preprocess(img, [1024, 1024])
        mask = self.preprocess(mask, [1024, 1024])

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }



