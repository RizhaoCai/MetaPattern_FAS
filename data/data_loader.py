import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import torch
import zip_dataset
from transforms import VisualTransform, get_augmentation_transforms
import torchvision.transforms as transforms
import logging

import pdb


def parse_data_list(data_list_path):
    csv = pd.read_csv(data_list_path, header=None)
    data_list = csv.get(0)
    face_labels = csv.get(1)

    return data_list, face_labels


def get_dataset_from_list(data_list_path, dataset_cls, transform, num_frames=1000, root_dir=''):

    data_file_list, face_labels = parse_data_list(data_list_path)

    num_file = data_file_list.size
    dataset_list = []

    for i in range(num_file):
        face_label = int(face_labels.get(i)==0) # 0 means real face and non-zero represents spoof
        file_path = data_file_list.get(i)

        zip_path = root_dir + file_path
        if not os.path.exists(zip_path):
            logging.warning("Skip {} (not exists)".format(zip_path))
            continue
        else:
            dataset = dataset_cls(zip_path, face_label, transform=transform, num_frames=num_frames)
            if len(dataset) == 0:
                logging.warning("Skip {} (zero elements)".format(zip_path))
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

    face_dataset_dir = '/home/rizhao/data/FAS/all_public_datasets_zip/EXT0.0/'



    transform =  transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((256,256)),
                transforms.ToTensor()
            ]
        )


    dataset_cls = zip_dataset.ZipDatasetPixelFPN

    test_dataset = get_dataset_from_list('data_list/debug.csv', dataset_cls, transform, root_dir=face_dataset_dir)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size, num_workers=num_workers,
                                                   shuffle=False, pin_memory=True, drop_last=True)

    data_iterator = iter(test_data_loader)

    data = data_iterator.next()
    import pdb; pdb.set_trace()