from torchvision import transforms
import torch
import logging

from data.prec.custom_transform import ShufflePatch

def get_augmentation_transforms(config):
    augmentation_transforms = []
    if config.TRAIN.AUG.ColorJitter.ENABLE:
        logging.info('Data Augmentation ColorJitter is ENABLED')
        brightness = config.TRAIN.AUG.ColorJitter.brightness
        contrast = config.TRAIN.AUG.ColorJitter.contrast
        saturation = config.TRAIN.AUG.ColorJitter.saturation
        hue = config.TRAIN.AUG.ColorJitter.hue
        augmentation_transforms.append(transforms.ColorJitter(brightness, contrast, saturation, hue))

    if config.TRAIN.AUG.RandomHorizontalFlip.ENABLE:
        logging.info('Data Augmentation RandomHorizontalFlip is ENABLED')
        p = config.TRAIN.AUG.RandomHorizontalFlip.p
        augmentation_transforms.append(transforms.RandomHorizontalFlip(p))

    if config.TRAIN.AUG.RandomCrop.ENABLE:
        logging.info('Data Augmentation RandomCrop is ENABLED')
        size = config.TRAIN.AUG.RandomCrop.size
        augmentation_transforms.append(transforms.RandomCrop(size))

    if config.TRAIN.AUG.RandomErasing.ENABLE:
        logging.info('Data Augmentation RandomErasing is ENABLED')
        p = config.TRAIN.AUG.RandomErasing.p
        scale = config.TRAIN.AUG.RandomErasing.scale
        ratio = config.TRAIN.AUG.RandomErasing.ratio
        augmentation_transforms.extend([transforms.ToTensor(), transforms.RandomErasing(p=p, scale=scale, ratio=ratio), transforms.ToPILImage()])

    if config.TRAIN.AUG.RandomErasing.ENABLE:
        logging.info('Data Augmentation RandomErasing is ENABLED')
        p = config.TRAIN.AUG.RandomErasing.p
        scale = config.TRAIN.AUG.RandomErasing.scale
        ratio = config.TRAIN.AUG.RandomErasing.ratio
        augmentation_transforms.extend([transforms.ToTensor(), transforms.RandomErasing(p=p, scale=scale, ratio=ratio), transforms.ToPILImage()])

    if config.TRAIN.AUG.ShufflePatch.ENABLE:
        logging.info('Data Augmentation RandomErasing is ENABLED')
        p = config.TRAIN.AUG.ShufflePatch.p
        size = config.TRAIN.AUG.ShufflePatch.size
        shuffle_patch_tranform = ShufflePatch(patch_size=size, shuffle_prob=p)
        augmentation_transforms.extend([shuffle_patch_tranform])

    return augmentation_transforms

class VisualTransform(torch.nn.Module):
    def __init__(self, config, augmentation_transforms=[]):
        super(VisualTransform, self).__init__()
        self.config = config
        img_size = config.DATA.IN_SIZE
        self.transform_list = [

            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size))
        ]
        self.transform_list.extend(augmentation_transforms)


        self.transform_list.extend(
            [
                transforms.ToTensor()
            ]
       )

        if self.config.DATA.NORMALIZE.ENABLE:
            norm_transform = transforms.Normalize(mean=self.config.DATA.NORMALIZE.MEAN, \
                                                  std=self.config.DATA.NORMALIZE.STD)
            self.transform_list.append(norm_transform)
            logging.info("Mean STD Normalize is ENABLED")

        self.transforms = transforms.Compose(
            self.transform_list
        )

    def forward(self, x):
        return self.transforms(x)

