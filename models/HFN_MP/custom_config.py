from yacs.config import CfgNode as CN

_C = CN()

# _C.OUTPUT_DIR = "output/meta_color/"
_C.OUTPUT_DIR = "/root/Desktop/workspace/meta-learning/FAS/MetaPattern_FAS/data/output/tmp"
_C.NORM_FLAG = True
_C.SEED = 666
_C.DEBUG = False
_C.DATA = CN()
_C.DATA.DATASET='ZipDataset'
_C.DATA.ROOT_DIR = "/root/Desktop/workspace/meta-learning/FAS/MetaPattern_FAS/data"

# _C.DATA.SUB_DIR = "EXT0.0"
_C.DATA.SUB_DIR = "FAS_data"
# _C.DATA.TRAIN_SRC_REAL_1 = 'data/data_list/CASIA-ALL-REAL.csv'
# _C.DATA.TRAIN_SRC_FAKE_1 = 'data/data_list/CASIA-ALL-FAKE.csv'

# _C.DATA.TRAIN_SRC_REAL_2 = 'data/data_list/MSU-MFSD-REAL.csv'
# _C.DATA.TRAIN_SRC_FAKE_2 = 'data/data_list/MSU-MFSD-FAKE.csv'


# _C.DATA.TRAIN_SRC_REAL_3 = 'data/data_list/REPLAY-ALL-REAL.csv'
# _C.DATA.TRAIN_SRC_FAKE_3 = 'data/data_list/REPLAY-ALL-FAKE.csv'

# _C.DATA.TARGET_DATA = 'data/data_list/OULU-NPU.csv'
_C.DATA.BATCH_SIZE = 32
_C.DATA.TRAIN_NF = 1000
_C.DATA.VAL_NF = 2
_C.DATA.TEST_NF = 2
_C.DATA.IN_SIZE = 256 # Input image size
_C.DATA.NORMALIZE = CN(new_allowed=True)
_C.DATA.NORMALIZE.ENABLE = False
_C.DATA.NORMALIZE.MEAN = [0.485, 0.456, 0.406]
_C.DATA.NORMALIZE.STD = [0.229, 0.224, 0.225]
_C.DATA.TEST = ""
_C.MODEL = CN()


_C.TRAIN = CN()
_C.TRAIN.INIT_LR = 0.01
_C.TRAIN.LR_EPOCH_1 = 0
_C.TRAIN.LR_EPOCH_2 = 150
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WEIGHT_DECAY = 0.0 # 5e-4
_C.TRAIN.WEIGHT_DECAY_T = 0.0 # ColorNet for TRANSFORMER
_C.TRAIN.MAX_ITER = 10# 1000000
_C.TRAIN.META_TRAIN_SIZE = 2
_C.TRAIN.ITER_PER_EPOCH = 10 #100
_C.TRAIN.META_PRE_TRAIN = True
_C.TRAIN.DROPOUT = 0.0
_C.TRAIN.EPOCHS = 10# 20
_C.TRAIN.SYNC_TRAINING = False
_C.TRAIN.IMAGENET_PRETRAIN = True
_C.TRAIN.RESUME = '' # Path to the resume ckpt
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

# TODO
_C.TRAIN.W_depth = 10
_C.TRAIN.W_metatest = 1
_C.TRAIN.META_LEARNING_RATE = 0.001
_C.TRAIN.BETAS = [0.9, 0.999]
_C.TRAIN.META_TEST_FREQ = 1
_C.TRAIN.VAL_FREQ = 1
_C.TRAIN.NUM_FRAMES = 1000
_C.TRAIN.INNER_LOOPS = 100
_C.TRAIN.RETRAIN_FROM_SCATCH = True


_C.TRAIN.OPTIM = 'SGD' # Adam

# ['None', 'CosineAnnealingLR']
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = ''
_C.TRAIN.LR_SCHEDULER.CosineAnnealingLR = CN()
_C.TRAIN.LR_SCHEDULER.CosineAnnealingLR.T_max = 400
_C.TRAIN.LR_SCHEDULER.CosineAnnealingLR.eta_min = 0.000001
_C.TRAIN.LR_SCHEDULER.CosineAnnealingLR.last_epoch = -1






_C.MODEL = CN()
_C.MODEL.IN_CHANNELS = 3
_C.MODEL.MEAN_STD_NORMAL = True

_C.MODEL.CHANNELS = CN()
_C.MODEL.CHANNELS.RGB = True
_C.MODEL.CHANNELS.HSV = False
_C.MODEL.CHANNELS.YCRCB = False
_C.MODEL.CHANNELS.YUV = False
_C.MODEL.CHANNELS.LAB = False
_C.MODEL.CHANNELS.XYZ = False

_C.TEST = CN()
_C.TEST.NUM_FRAMES = 2


def get_cfg_custom():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()