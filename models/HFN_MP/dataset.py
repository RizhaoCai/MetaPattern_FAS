import logging
import os
import zipfile

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms import ToTensor

from data.data_loader import parse_data_list
from data.transforms import VisualTransform
from data.transforms import get_augmentation_transforms
from data.zip_dataset import ZipDataset as _ZipDataset
import functools


class ZipDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, face_label, transforms=None, num_frames=1, train=True):
        self.folder_path = folder_path
        self.decode_flag = cv2.IMREAD_UNCHANGED
        self.train = train
        self.face_label = face_label

        if transforms is None:
            if not train:
                self.transforms = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transforms = T.Compose([
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transforms = transforms

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
        im = self.__read_image__(index)
        tensor = self.transforms(im)
        # print(tensor.shape)
        tensor = tensor.to(torch.float)
        target = {
            'face_label': self.face_label,
            'depth': self.face_label*torch.ones([1,32,32,], dtype=torch.float32)
        }
        return index, tensor, target, self.folder_path

    def __len__(self):
        return self.len



class ZipDataset(_ZipDataset):
    def __init__(self, folder_path, face_label, transform=None, num_frames=1, train=True):
        _ZipDataset.__init__(self, folder_path, face_label, transform=transform, num_frames=num_frames)

    def __getitem__(self, index):
        index, tensor, target, self.folder_path = _ZipDataset.__getitem__(self, index)
        target['depth'] =  self.face_label * torch.ones([1, 32, 32, ], dtype=torch.float32)
        return index, tensor, target, self.folder_path

class ZipDatasetPixelMC(ZipDataset):
    """
        ZipDataset with Pixel-wise target (Multi channel)
    """
    def __init__(self, folder_path, face_label, transform, num_frames=1000, config=None):
        super(ZipDatasetPixelMC, self).__init__(folder_path, face_label, transform, num_frames)

        self.config = config
    def __getitem__(self, index):
        im = self.__read_image__(index)
        pixel_maps_size = 32

        pixel_maps_size = [32, 16, 8, 4, 2]
        pixel_maps = []
        for s in pixel_maps_size:
            pixel_maps.append(self.face_label * torch.ones([s, s]))

        channels = channel_list(im, self.config)
        channels = [ self.transform(im) for im in channels ]

        channels = torch.cat(channels, 0)
        target = {
            'face_label': self.face_label,
            'pixel_maps': pixel_maps
        }
        return index, channels, target, self.folder_path


# def channel_list(bgr_im, config=None):
#     channel_list = []
#     if config.MODEL.CHANNELS.RGB:
#         channel_list.append(bgr_im)
#     if config.MODEL.CHANNELS.HSV:
#         img_hsv = cv2.cvtColor(bgr_im, cv2.COLOR_BGR2HSV)
#         channel_list.append(img_hsv)

#     if config.MODEL.CHANNELS.YCRCB:
#         img_ycrcb = cv2.cvtColor(bgr_im, cv2.COLOR_BGR2YCrCb)

#         channel_list.append(img_ycrcb)

#     if config.MODEL.CHANNELS.LAB:
#         lab_image = cv2.cvtColor(bgr_im, cv2.COLOR_BGR2LAB)
#         channel_list.append(lab_image)

#     if config.MODEL.CHANNELS.YUV:
#         yuv_image = cv2.cvtColor(bgr_im, cv2.COLOR_BGR2YUV)
#         channel_list.append(yuv_image)

#     if config.MODEL.CHANNELS.XYZ:
#         xyz_image = cv2.cvtColor(bgr_im, cv2.COLOR_BGR2XYZ)
#         channel_list.append(xyz_image)

#     return channel_list



def channel_list(bgr_im, config=None):
    """
    Convert and return a list of image channels as Tensors.
    """
    channel_list = []
    to_tensor = ToTensor()  # Convert PIL.Image to Tensor

    # Ensure input is a NumPy array if not already
    if isinstance(bgr_im, Image.Image):
        bgr_im = np.array(bgr_im)  # Convert PIL.Image.Image to NumPy array

    if config.MODEL.CHANNELS.RGB:
        rgb_image = cv2.cvtColor(bgr_im, cv2.COLOR_BGR2RGB)
        channel_list.append(to_tensor(Image.fromarray(rgb_image)))
    if config.MODEL.CHANNELS.HSV:
        img_hsv = cv2.cvtColor(bgr_im, cv2.COLOR_BGR2HSV)
        channel_list.append(to_tensor(Image.fromarray(img_hsv)))
    if config.MODEL.CHANNELS.YCRCB:
        img_ycrcb = cv2.cvtColor(bgr_im, cv2.COLOR_BGR2YCrCb)
        channel_list.append(to_tensor(Image.fromarray(img_ycrcb)))
    if config.MODEL.CHANNELS.LAB:
        lab_image = cv2.cvtColor(bgr_im, cv2.COLOR_BGR2LAB)
        channel_list.append(to_tensor(Image.fromarray(lab_image)))
    if config.MODEL.CHANNELS.YUV:
        yuv_image = cv2.cvtColor(bgr_im, cv2.COLOR_BGR2YUV)
        channel_list.append(to_tensor(Image.fromarray(yuv_image)))
    if config.MODEL.CHANNELS.XYZ:
        xyz_image = cv2.cvtColor(bgr_im, cv2.COLOR_BGR2XYZ)
        channel_list.append(to_tensor(Image.fromarray(xyz_image)))

    return channel_list

def get_dataset_from_list(data_list_path, dataset_cls, train=True, transform=None, num_frames=1, root_dir=''):
    data_file_list, face_labels = parse_data_list(data_list_path)
    dataset_list = []
    for i, file_entry in enumerate(data_file_list):
        # Split file entry to extract the actual file path
        file_path = file_entry.split(',')[0]
        folder_path = os.path.dirname(file_path)  # Extract the folder containing the file
        total_path = os.path.abspath(folder_path)  # Use the absolute path
        
        # Debugging: Print paths for verification
        # print(f"Processing file: {file_path}")
        # print(f"Total folder path: {total_path}")

        if not os.path.exists(total_path):
            print(f"Skip {total_path} (not exists)")
            continue

        # Create the dataset object
        face_label = int(face_labels[i] == 0)  # 0 means real face, non-zero is spoof
        try:
            dataset = dataset_cls(total_path, face_label, transform, num_frames=num_frames)
        except Exception as e:
            print(f"Error creating dataset for {total_path}: {e}")
            continue

        if len(dataset) == 0:
            print(f"Skip {total_path} (zero elements)")
            continue
        else:
            dataset_list.append(dataset)

    if not dataset_list:
        print("No valid datasets found.")
        return None

    # Concatenate all datasets
    final_dataset = torch.utils.data.ConcatDataset(dataset_list)
    return final_dataset

def get_data_loader(config):


    aug_transform = get_augmentation_transforms(config)
    train_data_transform = VisualTransform(config, aug_transform)
    test_data_transform = VisualTransform(config)
    root_dir = os.path.join(config.DATA.ROOT_DIR, config.DATA.SUB_DIR)
    Dataset = functools.partial(ZipDatasetPixelMC, config=config)

    batch_size = config.DATA.BATCH_SIZE
    if config.DATA.TEST:
        tgt_test_data = get_dataset_from_list(config.DATA.TEST, Dataset, transform=test_data_transform,
                                              root_dir=root_dir, num_frames=config.TEST.NUM_FRAMES)


        tgt_dataloader = DataLoader(tgt_test_data, batch_size=batch_size, shuffle=False, drop_last=True)
        return tgt_dataloader
    src1_train_data_fake_list = config.DATA.TRAIN_SRC_FAKE_1
    src1_train_data_real_list = config.DATA.TRAIN_SRC_REAL_1

    src2_train_data_fake_list = config.DATA.TRAIN_SRC_FAKE_2
    src2_train_data_real_list = config.DATA.TRAIN_SRC_REAL_2

    src3_train_data_fake_list = config.DATA.TRAIN_SRC_FAKE_3
    src3_train_data_real_list = config.DATA.TRAIN_SRC_REAL_3

    num_frames = config.TRAIN.NUM_FRAMES
    src1_train_data_fake = get_dataset_from_list(src1_train_data_fake_list, Dataset, transform=train_data_transform, num_frames=config.TRAIN.NUM_FRAMES, root_dir=root_dir)
    src1_train_data_real = get_dataset_from_list(src1_train_data_real_list, Dataset, transform=train_data_transform, num_frames=config.TRAIN.NUM_FRAMES, root_dir=root_dir)
    src2_train_data_fake = get_dataset_from_list(src2_train_data_fake_list, Dataset, transform=train_data_transform, num_frames=config.TRAIN.NUM_FRAMES, root_dir=root_dir)
    src2_train_data_real = get_dataset_from_list(src2_train_data_real_list, Dataset, transform=train_data_transform, num_frames=config.TRAIN.NUM_FRAMES, root_dir=root_dir)
    src3_train_data_fake = get_dataset_from_list(src3_train_data_fake_list, Dataset, transform=train_data_transform, num_frames=config.TRAIN.NUM_FRAMES, root_dir=root_dir)
    src3_train_data_real = get_dataset_from_list(src3_train_data_real_list, Dataset, transform=train_data_transform, num_frames=config.TRAIN.NUM_FRAMES, root_dir=root_dir)



    tgt_test_data_list = config.DATA.TARGET_DATA
    tgt_test_data = get_dataset_from_list(tgt_test_data_list, Dataset, transform=test_data_transform, root_dir=root_dir, num_frames = config.TEST.NUM_FRAMES)


    # print('Load Target Data')

    src1_train_dataloader_fake = DataLoader(src1_train_data_fake,
                                            batch_size=batch_size, shuffle=True)
    src1_train_dataloader_real = DataLoader(src1_train_data_real,
                                            batch_size=batch_size, shuffle=True)
    src2_train_dataloader_fake = DataLoader(src2_train_data_fake,
                                            batch_size=batch_size, shuffle=True)
    src2_train_dataloader_real = DataLoader(src2_train_data_real,
                                            batch_size=batch_size, shuffle=True)
    src3_train_dataloader_fake = DataLoader(src3_train_data_fake,
                                            batch_size=batch_size, shuffle=True)
    src3_train_dataloader_real = DataLoader(src3_train_data_real,
                                            batch_size=batch_size, shuffle=True)
    tgt_dataloader = DataLoader(tgt_test_data, batch_size=batch_size, shuffle=False, drop_last=True)
    return src1_train_dataloader_fake, src1_train_dataloader_real, \
           src2_train_dataloader_fake, src2_train_dataloader_real, \
           src3_train_dataloader_fake, src3_train_dataloader_real, \
           tgt_dataloader