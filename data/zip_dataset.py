# coding: utf-8
# import torch.multiprocessing as mp
# mp.set_start_method('spawn')

import cv2
import numpy as np
import torch
import zipfile
from torch.utils.data import Dataset
import os
from PIL import Image


class ZipDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, face_label, transform, num_frames=1000):
        self.folder_path = folder_path
        self.transform = transform
        self.decode_flag = cv2.IMREAD_UNCHANGED
        self.face_label = face_label
        self.image_list = []
        exts = ['png', 'jpg']
        for ext in exts:
            self.image_list += list(filter(lambda x: x.lower().endswith(ext), os.listdir(folder_path)))

        if len(self.image_list) > 0:
            self.image_list = self.image_list[:num_frames]
        self.len = len(self.image_list)
    
    def __read_image__(self, index):
        image_name = self.image_list[index]
        image_path = os.path.join(self.folder_path, image_name)
        
        im = cv2.imread(image_path, self.decode_flag) 
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (256, 256))
        im = Image.fromarray(im)
        return im  

    def __getitem__(self, index):
        im = self.__read_image__(index) # cv2 image, format [H, W, C], BGR
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # im = im.transpose((2,0,1))
        tensor = self.transform(im)
        tensor = tensor.to(torch.float)
        target = {
            'face_label':  self.face_label
        }
        return index, tensor, target, self.folder_path

    def __len__(self):
        return self.len


class ZipDatasetPixelFPN(ZipDataset):
    """
        ZipDataset with with Pixel target for FPN
    """

    def __init__(self, folder_path, face_label, transform, num_frames=1000):
        super(ZipDatasetPixelFPN, self).__init__(folder_path, face_label, transform, num_frames)


    def __getitem__(self, index):

        im = self.__read_image__(index)
        # No RGB to BGR
        pixel_maps_size = [32, 16, 8, 4, 2]
        pixel_maps = []
        for s in pixel_maps_size:
            pixel_maps.append(self.face_label*torch.ones([s,s]))
        im = self.transform(im)
        target = {
            'face_label': self.face_label,
            'pixel_maps': pixel_maps
        }
        return index, im, target, self.folder_path


class ZipDatasetPixel(ZipDataset):
    """
        ZipDataset with Pixel-wise target
    """

    def __init__(self, folder_path, face_label, transform, num_frames=1000):
        super(ZipDatasetPixel, self).__init__(folder_path, face_label, transform, num_frames)


    def __getitem__(self, index):

        im = self.__read_image__(index)
        pixel_maps_size = 32

        pixel_maps = self.face_label*torch.ones([pixel_maps_size,pixel_maps_size])
        im = self.transform(im)
        target = {
            'face_label': self.face_label,
            'pixel_maps': pixel_maps
        }
        return index, im, target, self.folder_path


class ZipDatasetMultiChannel(ZipDataset):
    """
        ZipDatasetMultiChannel: RGB+HSV or RFB+YUV
    """

    def __init__(self, config, folder_path, face_label, transform, num_frames=1000):
        super(ZipDatasetMultiChannel, self).__init__(folder_path, face_label, transform, num_frames)


    def __getitem__(self, index):

        im = self.__read_image__(index)
        im_rfb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        im = self.transform(im)

        return index, im, target, self.folder_path

    # def _extract_lbp_()