# coding: utf-8
# import torch.multiprocessing as mp
# mp.set_start_method('spawn')

import cv2
import numpy as np
import torch
import zipfile
from torch.utils.data import Dataset



class ZipDataset(torch.utils.data.Dataset):
    def __init__(self, zip_file_path, face_label, transform, num_frames=1000):
        self.zip_file_path = zip_file_path
        self.transform = transform
        self.decode_flag = cv2.IMREAD_UNCHANGED
        self.face_label = face_label

        self.image_list_in_zip = []
        with zipfile.ZipFile(self.zip_file_path, "r") as zip:
            lst = zip.namelist()
            exts = ['png', 'jpg']
            for ext in exts:
                self.image_list_in_zip += list(filter(lambda x: x.lower().endswith(ext), lst))

        if len(self.image_list_in_zip) > num_frames:
            sample_indices = np.linspace(0, len(self.image_list_in_zip)-1, num=num_frames, dtype=int)
            self.image_list_in_zip = [self.image_list_in_zip[id] for id in sample_indices]

        self.len = len(self.image_list_in_zip)

    def __read_image_from_zip__(self, index):
        image_name_in_zip = self.image_list_in_zip[index]
        with zipfile.ZipFile(self.zip_file_path, "r") as zip:
            bytes_ = zip.read(image_name_in_zip)
            bytes_ = np.frombuffer(bytes_, dtype=np.uint8)
            im = cv2.imdecode(bytes_, self.decode_flag)  # cv2 image
            return im


    def __getitem__(self, index):
        im = self.__read_image_from_zip__(index) # cv2 image, format [H, W, C], BGR
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # im = im.transpose((2,0,1))
        tensor = self.transform(im)
        tensor = tensor.to(torch.float)
        target = {
            'face_label':  self.face_label
        }
        return index, tensor, target, self.zip_file_path

    def __len__(self):
        return self.len


class ZipDatasetPixelFPN(ZipDataset):
    """
        ZipDataset with with Pixel target for FPN
    """

    def __init__(self, zip_file_path, face_label, transform, num_frames=1000):
        super(ZipDatasetPixelFPN, self).__init__(zip_file_path, face_label, transform, num_frames)


    def __getitem__(self, index):

        im = self.__read_image_from_zip__(index)
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
        return index, im, target, self.zip_file_path


class ZipDatasetPixel(ZipDataset):
    """
        ZipDataset with Pixel-wise target
    """

    def __init__(self, zip_file_path, face_label, transform, num_frames=1000):
        super(ZipDatasetPixel, self).__init__(zip_file_path, face_label, transform, num_frames)


    def __getitem__(self, index):

        im = self.__read_image_from_zip__(index)
        pixel_maps_size = 32

        pixel_maps = self.face_label*torch.ones([pixel_maps_size,pixel_maps_size])
        im = self.transform(im)
        target = {
            'face_label': self.face_label,
            'pixel_maps': pixel_maps
        }
        return index, im, target, self.zip_file_path


class ZipDatasetMultiChannel(ZipDataset):
    """
        ZipDatasetMultiChannel: RGB+HSV or RFB+YUV
    """

    def __init__(self, config, zip_file_path, face_label, transform, num_frames=1000):
        super(ZipDatasetMultiChannel, self).__init__(zip_file_path, face_label, transform, num_frames)


    def __getitem__(self, index):

        im = self.__read_image_from_zip__(index)
        im_rfb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        im = self.transform(im)

        return index, im, target, self.zip_file_path

    # def _extract_lbp_()