# Copyright (c) Youngwan Lee (ETRI) All Rights Reserved.
from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------#
# Init
# -----------------------------------------------------------------------------#
_C = CN(new_allowed=True)
_C.OUTPUT_DIR = "output/tmp"
_C.DEBUG = False
_C.SEED = None
_C.CUDA = True
_C.NOTES = '' # Any comments. Can help remember what the experiment is about.





_C.MODEL = CN(new_allowed=True)
_C.DATA = CN(new_allowed=True)
_C.TRAIN = CN(new_allowed=True)
_C.TEST = CN(new_allowed=True)

# ---------------------------------------------------------------------------- #
# Data config
# ---------------------------------------------------------------------------- #
_C.DATA.TRAIN = ""
_C.DATA.VAL = ""
_C.DATA.TEST = ""
_C.DATA.IN_SIZE = 256 # Input image size
_C.DATA.DATASET = "ZipDataset"
_C.DATA.ROOT_DIR = "/home/rizhao/data/FAS/MTCNN/"
_C.DATA.SUB_DIR = "align"
_C.DATA.NUM_FRAMES = 1000 # number of frames extracted from a video
_C.DATA.BATCH_SIZE = 16
_C.DATA.NUM_WORKERS = 4

_C.DATA.NORMALIZE = CN(new_allowed=True)
_C.DATA.NORMALIZE.ENABLE = False
_C.DATA.NORMALIZE.MEAN = [0.485, 0.456, 0.406]
_C.DATA.NORMALIZE.STD = [0.229, 0.224, 0.225]




# ---------------------------------------------------------------------------- #
# MODEL/NETWORK  config
# ---------------------------------------------------------------------------- #
_C.MODEL.NUM_CLASSES = 2

# ---------------------------------------------------------------------------- #
# MODEL/NETWORK  config
# ---------------------------------------------------------------------------- #
_C.TRAIN.RESUME = '' # Path to the resume ckpt
_C.TRAIN.INIT_LR = 1e-4
_C.TRAIN.WEIGHT_DECAY = 0.0
_C.TRAIN.EPOCHS = 20
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.LR_PATIENCE = 0
_C.TRAIN.PATIENCE = 100
_C.TRAIN.SAVE_BEST = True # Only save the best model while training
_C.TRAIN.PRINT_FREQ = 1000
_C.TRAIN.VAL_FREQ = 1
_C.TRAIN.NUM_FRAMES = 1000


_C.TRAIN.AUG = CN(new_allowed=True) # Augmentation. For training only
_C.TRAIN.AUG.ColorJitter = CN(new_allowed=True)
_C.TRAIN.AUG.ColorJitter.ENABLE=False
_C.TRAIN.AUG.ColorJitter.brightness = 0.25
_C.TRAIN.AUG.ColorJitter.contrast = 0.5
_C.TRAIN.AUG.ColorJitter.hue = 0
_C.TRAIN.AUG.ColorJitter.saturation = 0


_C.TRAIN.AUG.RandomHorizontalFlip = CN(new_allowed=True)
_C.TRAIN.AUG.RandomHorizontalFlip.ENABLE = False
_C.TRAIN.AUG.RandomHorizontalFlip.p = 0.5


_C.TRAIN.AUG.RandomCrop = CN(new_allowed=True)
_C.TRAIN.AUG.RandomCrop.ENABLE = False
_C.TRAIN.AUG.RandomCrop.size = 256

_C.TRAIN.AUG.RandomErasing = CN(new_allowed=True)
_C.TRAIN.AUG.RandomErasing.ENABLE = False
_C.TRAIN.AUG.RandomErasing.p = 0.5
_C.TRAIN.AUG.RandomErasing.scale = (0.02, 0.33)
_C.TRAIN.AUG.RandomErasing.ratio = (0.3, 3.3)

_C.TRAIN.AUG.ShufflePatch = CN(new_allowed=True)
_C.TRAIN.AUG.ShufflePatch.ENABLE = False
_C.TRAIN.AUG.ShufflePatch.p = 0.5
_C.TRAIN.AUG.ShufflePatch.size = 32



# TEST Config
_C.TEST.CKPT = '' # checkpoint to load
_C.TEST.TAG = 'Default'
_C.TEST.NO_INFERENCE = False # Load metrics from TEST.OUTPUT_DIR and conduct testing
_C.TEST.THR = 0.5 # Threshold for calculating HTER
_C.TEST.MORE = False # Whether to collect more statistics
_C.TEST.NUM_FRAMES = 1000



def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()