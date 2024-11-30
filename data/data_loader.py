import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import torch
import zip_dataset
from transforms import VisualTransform, get_augmentation_transforms
import torchvision.transforms as transforms
import logging
import cv2

import pdb

def make_data_list(data_path):
    """
    1.FAS_data 하위의 모든 jpg 파일들을 아래와 같이 분류 합니다.
      test = TEST
      dev,train = TRAIN
      live = REAL
      spoof = FAKE
    2.data_list 파일들의 내용은 아래와 같은 포맷으로 저장 됩니다.
      file_path,face_label,height, width
    3.예제의 ._로 시작하는 파일에는 오류를 발생합니다(이미지포맷이 지원하지 않음)
      만일 ._ 파일을 사용해야하는 경우 주석을 참고하여 2개의 라인을 삭제하세요.
    """
    root_dir = "FAS_data"
    for root, _, files in os.walk(root_dir):  # search all directories under the root_dir
        for file in files:
            if file.endswith(".jpg"):
                file_path = os.path.join(root, file) # current image file path
                folders = file_path.split('/') # split char = [Window : \\ , Others : /]
                height=0
                width=0
                channels=0

                # For _. files, please remove the following two lines.
                img = cv2.imread(file_path)
                height, width, channels = img.shape

                if any(word in file_path.lower() for word in ["test"]): 
                    if "live" in file_path.lower():   
                        list_file = folders[1] + "-TEST-REAL.txt"
                        face_labe="0"
                    elif "spoof" in file_path.lower():
                        list_file = folders[1] + "-TEST-FAKE.txt"
                        face_labe="1"
                elif any(word in file_path.lower() for word in ["train","dev"]): 
                    if "live" in file_path.lower():   
                        list_file = folders[1] + "-TRAIN-REAL.txt"
                        face_labe="0"
                    elif "spoof" in file_path.lower():
                        list_file = folders[1] + "-TRAIN-FAKE.txt"
                        face_labe="1"

                with open("data_list/"+list_file, "a") as f:
                    f.write(file_path + "," + face_labe + "," + str(height) + "," + str(width) +"\n")     
    return True


def parse_data_list(data_list_path):
    data_file_list = []
    face_labels = []
    
    with open(data_list_path, 'r') as f:
        lines = f.readlines()


    for line in lines:
        line = line.strip()  # 줄바꿈 제거
        if not line:  # 빈 줄 건너뜀
            continue

        file_path = line
        face_label = 0 if 'live' in file_path.lower() else 1  # 'live'가 포함되면 0, 아니면 1

        data_file_list.append(file_path)
        face_labels.append(face_label)

    return data_file_list, face_labels


def get_dataset_from_list(data_list_path, dataset_cls, transform, num_frames=1000, root_dir=''):

    data_file_list, face_labels = parse_data_list(data_list_path)

    num_file = data_file_list.size
    dataset_list = []

    for i in range(num_file):
        face_label = int(face_labels.get(i)==0) # 0 means real face and non-zero represents spoof
        file_path = data_file_list.get(i)
        
        total_path = os.path.join(file_path, root_dir)
        if not os.path.exists(total_path):
            logging.warning("Skip {} (not exists)".format(total_path))
            continue
        else:
            dataset = dataset_cls(total_path, face_label, transform=transform, num_frames=num_frames)
            if len(dataset) == 0:
                logging.warning("Skip {} (zero elements)".format(total_path))
                continue
            else:
                dataset_list.append(dataset)
    final_dataset = torch.utils.data.ConcatDataset(dataset_list)
    return final_dataset

def get_data_loader(config):
    batch_size = config.DATA.BATCH_SIZE
    num_workers = 0 if config.DEBUG else config.DATA.NUM_WORKERS
    dataset_cls = zip_dataset.__dict__[config.DATA.DATASET]
    dataset_root_dir = config.DATA.ROOT_DIR
    dataset_subdir = config.DATA.SUB_DIR  # 'EXT0.2'
    face_dataset_dir = os.path.join(dataset_root_dir, dataset_subdir)
    num_frames_train = config.TRAIN.NUM_FRAMES
    num_frames_test = config.TEST.NUM_FRAMES

    assert config.DATA.TRAIN or config.DATA.TEST, "Please provide at least a data_list"

    test_data_transform = VisualTransform(config)
    if config.DATA.TEST:

        test_dataset =get_dataset_from_list(config.DATA.TEST, dataset_cls, test_data_transform, num_frames_test, root_dir=face_dataset_dir)
        test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size, num_workers=num_workers,
                                                       shuffle=False, pin_memory=True, drop_last=True)
        return test_data_loader
    else:
        assert config.DATA.TRAIN, "CONFIG.DATA.TRAIN should be provided"

        aug_transform = get_augmentation_transforms(config)
        train_data_transform = VisualTransform(config, aug_transform)


        train_dataset = get_dataset_from_list(config.DATA.TRAIN, dataset_cls, train_data_transform, num_frames_train, root_dir=face_dataset_dir)
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size, num_workers=num_workers,
                                                            shuffle=True, pin_memory=True, drop_last=True)

        assert config.DATA.VAL, "CONFIG.DATA.VAL should be provided"
        val_dataset =get_dataset_from_list(config.DATA.VAL, dataset_cls, test_data_transform, num_frames=num_frames_test, root_dir=face_dataset_dir)
        val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size, num_workers=num_workers,
                                                       shuffle=False, pin_memory=True, drop_last=True)


        return train_data_loader, val_data_loader

if __name__ == '__main__':
    batch_size = 4
    num_workers = 2
    make_data_list('FAS_data')
    # face_dataset_dir = '/home/rizhao/data/FAS/all_public_datasets_zip/EXT0.0/'
    # face_dataset_dir = 'FAS_data'


    # transform =  transforms.Compose(
    #         [
    #             transforms.ToPILImage(),
    #             transforms.Resize((256,256)),
    #             transforms.ToTensor()
    #         ]
    #     )


    # dataset_cls = zip_dataset.ZipDatasetPixelFPN

    # test_dataset = get_dataset_from_list('debug.csv', dataset_cls, transform, root_dir=face_dataset_dir)
    # test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size, num_workers=num_workers,
    #                                                shuffle=False, pin_memory=True, drop_last=True)

    # data_iterator = iter(test_data_loader)

    # data = data_iterator.next()
    # import pdb; pdb.set_trace()