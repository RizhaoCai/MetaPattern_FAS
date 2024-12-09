import logging
import os
import random

import torch
import torch.optim as optim
from tqdm import tqdm

from models.base import BaseTrainer
from utils.utils import AverageMeter
from .dataset import get_data_loader
from .network import HierachicalFusionNetwork
from .network import PatternExtractor, get_state_dict
from itertools import chain

#try:
#    from torch.utils.tensorboard import SummaryWriter
#except:
# from tensorboardX import SummaryWriter

from torchvision.transforms import Normalize
import sys

logging.basicConfig(
            level=logging.INFO,  # Set to INFO or DEBUG to see the messages
            format='%(asctime)s - %(levelname)s - %(message)s',  # Format for clarity
            datefmt='%Y-%m-%d %H:%M:%S',  # Optional: timestamp
            handlers=[
                        logging.StreamHandler(sys.stdout),  # Logs to console
                        logging.FileHandler("./logging.log")  # Logs to file
                    ]
        )


class Trainer(BaseTrainer):
    """
    Trainer encapsulates all the logic necessary for
    training the Recurrent Attention Model.

    All hyperparameters are provided by the user in the
    config file.
    """

    def __init__(self, config):
        """
        Construct a new Trainer instance.
        Args
        ----
        - config: object containing command line arguments.
        - data_loader: data iterator
        """
        super(Trainer, self).__init__(config)
        self.config = config

        # Initilization


        self.pattern_extractor = PatternExtractor().cuda()
        self.hfn = HierachicalFusionNetwork(mean_std_normalize=self.config.MODEL.MEAN_STD_NORMAL,
                                  dropout_rate=self.config.TRAIN.DROPOUT).cuda()
    def get_dataloader(self):
        return get_data_loader(self.config)

    def train(self):

        if self.config.TRAIN.SYNC_TRAINING:
            self.sync_training()
            return
        if self.config.TRAIN.META_PRE_TRAIN:
            self.meta_train()

        self.train_hfn_from_scratch()

    def load_checkpoint(self, ckpt_path):

        logging.info("[*] Loading model from {}".format(ckpt_path))

        ckpt = torch.load(ckpt_path)
        # load variables from checkpoint
        self.start_epoch = ckpt['epoch']
        self.global_step = ckpt['global_step']
        self.valid_metric = ckpt['val_metrics']
        # self.best_valid_acc = ckpt['best_valid_acc']
        self.hfn.load_state_dict(ckpt['model_state'][1])
        self.pattern_extractor.load_state_dict(ckpt['model_state'][0])

        if hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(ckpt['optim_state'])

        logging.info(
            "[*] Loaded {} checkpoint @ epoch {}".format(
                ckpt_path, ckpt['epoch'])
        )

    def meta_train(self, hfn=None, pattern_extractor=None):
        """

            Meta-train and test:
                Several things to determine:
                    - How many meta-train iterations for one-time meta-test?
                    - How many iterations to reshuffle the domain data-loaders

                # Regulation loss
        """
        # Get hyperparameters
        meta_learning_rate = self.config.TRAIN.META_LEARNING_RATE
        betas = self.config.TRAIN.BETAS
        max_iter = self.config.TRAIN.MAX_ITER
        val_freq = self.config.TRAIN.VAL_FREQ
        meta_test_freq = self.config.TRAIN.META_TEST_FREQ
        metatrainsize = self.config.TRAIN.META_TRAIN_SIZE  # 2
        # Get network

        # self.pattern_extractor.train()
        # Pretrain?
        if hfn is not None:
            self.hfn = hfn
        if pattern_extractor is not None:
            self.pattern_extractor = pattern_extractor

        # if self.config.TRAIN.IMAGENET_PRETRAIN:
            
        #     pretrain_model_path = 'models/HFN_MP/hfn_pretrain.pth'
        #     logging.info("Loading ImageNet Pretrain")
        #     if not os.path.exists(pretrain_model_path):
        #         get_state_dict() 
        #     imagenet_pretrain = torch.load(pretrain_model_path)
        #     self.hfn.load_state_dict(imagenet_pretrain, strict=False)

        # self.hfn.train()
        self.pattern_extractor.cuda()

        # criterionCls = nn.CrossEntropyLoss()
        criterionDepth = torch.nn.MSELoss()
        criterionReconstruction = torch.nn.MSELoss()
        criterionCLS = torch.nn.CrossEntropyLoss()

        if self.config.TRAIN.OPTIM == 'Adam':
            self.optimizer_fpn = optim.Adam(
                self.hfn.parameters(),
                lr=meta_learning_rate,
                # betas=betas
            )

            self.optimizer_color = optim.Adam(
                self.pattern_extractor.parameters(),
                lr=meta_learning_rate * self.config.TRAIN.INNER_LOOPS,
                # betas=betas
            )

        elif self.config.TRAIN.OPTIM == 'SGD':
            self.optimizer_fpn = optim.SGD(
                self.hfn.parameters(),
                lr=meta_learning_rate,
                momentum=0.9,
                weight_decay = self.config.TRAIN.WEIGHT_DECAY
            )

            self.optimizer_color = optim.SGD(
                self.pattern_extractor.parameters(),
                lr=meta_learning_rate * self.config.TRAIN.INNER_LOOPS,
                # betas=betas
                momentum=0.9,
                weight_decay = self.config.TRAIN.WEIGHT_DECAY_T
            )

        else:
            raise NotImplementedError

        # Add LR scheduler
        self.lr_scheduler = None
        if self.config.TRAIN.LR_SCHEDULER.NAME=='CosineAnnealingLR': # Default ''
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_fpn,
                T_max = self.config.TRAIN.LR_SCHEDULER.CosineAnnealingLR.T_max,
                eta_min = self.config.TRAIN.LR_SCHEDULER.CosineAnnealingLR.eta_min,
                last_epoch=self.config.TRAIN.LR_SCHEDULER.CosineAnnealingLR.last_epoch,
        )

        # tensorboard_dir = os.path.join(self.config.OUTPUT_DIR, "tensorboard")
        # os.makedirs(tensorboard_dir, exist_ok=True)
        # self.tensorboard = None if self.config.DEBUG else SummaryWriter(tensorboard_dir)

        src1_train_dataloader_fake, src1_train_dataloader_real, \
        src2_train_dataloader_fake, src2_train_dataloader_real, \
        src3_train_dataloader_fake, src3_train_dataloader_real, \
        tgt_valid_dataloader = self.get_dataloader()

        if self.config.TRAIN.RESUME and os.path.exists(self.config.TRAIN.RESUME):
            logging.info("Resume=True.")
            self.load_checkpoint(self.config.TRAIN.RESUME)

        epoch = 1

        iter_per_epoch = self.config.TRAIN.ITER_PER_EPOCH
        src1_train_dataloader_real = {
            "data_loader": src1_train_dataloader_real,
            "iterator": iter(src1_train_dataloader_real)
        }
        src2_train_dataloader_real = {
            "data_loader": src2_train_dataloader_real,
            "iterator": iter(src2_train_dataloader_real)
        }
        src3_train_dataloader_real = {
            "data_loader": src3_train_dataloader_real,
            "iterator": iter(src3_train_dataloader_real)
        }

        real_loaders = [src1_train_dataloader_real, src2_train_dataloader_real, src3_train_dataloader_real]

        src1_train_dataloader_fake = {
            "data_loader": src1_train_dataloader_fake,
            "iterator": iter(src1_train_dataloader_fake)
        }
        src2_train_dataloader_fake = {
            "data_loader": src2_train_dataloader_fake,
            "iterator": iter(src2_train_dataloader_fake)
        }
        src3_train_dataloader_fake = {
            "data_loader": src3_train_dataloader_fake,
            "iterator": iter(src3_train_dataloader_fake)
        }

        fake_loaders = [src1_train_dataloader_fake, src2_train_dataloader_fake, src3_train_dataloader_fake]

        pbar = tqdm(range(1, max_iter + 1), ncols=160)
        loss_meta_test = 0
        loss_meta_train = 0

        mean_std_normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        for iter_num in pbar:
            if (iter_num != 0 and iter_num % iter_per_epoch == 0):
                epoch = epoch + 1
                logging.info("Training at Epoch {}".format(epoch))

            domain_list = [0, 1, 2]  # 3 src domains # TODO: Remove hard code on future

            random.shuffle(domain_list)
            meta_train_list = domain_list[:metatrainsize]
            meta_test_list = domain_list[metatrainsize:]
            meta_train_loader_list_real = [real_loaders[i] for i in meta_train_list]
            meta_train_loader_list_fake = [fake_loaders[i] for i in meta_train_list]

            meta_test_loader_list_real = [real_loaders[i] for i in meta_test_list]
            meta_test_loader_list_fake = [fake_loaders[i] for i in meta_test_list]

            # ================ Meta-test =====================================
            # ================ Inner update loop =============================
            # ================ Update
            lambda_rec = 0
            inner_num = self.config.TRAIN.INNER_LOOPS
            for j in range(inner_num):
                image_meta_test, label_meta_test, depth_meta_test = get_data_from_loader_list(
                    meta_test_loader_list_real, meta_test_loader_list_fake)

                image_meta_test = image_meta_test.cuda()
                label_meta_test = label_meta_test.cuda()
                depth_meta_test = [d.cuda() for d in depth_meta_test] #.cuda()

                img_colored, reconstruct_rgb = self.pattern_extractor(image_meta_test)
                # TODO: Get regulation loss for the output img_color, such as MMD, or PCA, KL Divergence
                reconstruction_loss = criterionReconstruction(reconstruct_rgb, image_meta_test)

                depth_pred, cls_preds = self.hfn(image_meta_test, img_colored)  # TODO

                mse_loss = criterionDepth(depth_pred[0].squeeze(), depth_meta_test[0])
                cls_loss = criterionCLS(cls_preds[0], label_meta_test.cuda())
                loss_color_net = mse_loss + cls_loss + lambda_rec * reconstruction_loss
                self.optimizer_color.zero_grad()
                loss_color_net.backward()
                self.optimizer_color.step()

                loss_meta_test = loss_color_net.item()

            del image_meta_test, label_meta_test, depth_meta_test, img_colored
            # ================ Meta-train =======================
            # When implementing meta-train, the optimizer does not call step()
            # Only gradients are calculated and used for adjusting the gradients from meta-test.
            # The adjusted gradients are then BP is done.

            image_meta_train, label_meta_train, depth_meta_train = get_data_from_loader_list(
                meta_train_loader_list_real,
                meta_train_loader_list_fake)
            image_meta_train = image_meta_train.cuda()
            label_meta_train = label_meta_train.cuda()
            depth_meta_train = [d.cuda() for d in depth_meta_train] #.cuda() 

            img_colored, _ = self.pattern_extractor(image_meta_train)

            map_pred_outs, cls_preds = self.hfn(image_meta_train, img_colored)

            mse_loss = criterionDepth(map_pred_outs[0].squeeze(), depth_meta_train[0])
            cls_loss = criterionCLS(cls_preds[0], label_meta_train)

            loss_fpn = mse_loss + cls_loss
            loss_fpn = loss_fpn / 2
            loss_meta_train = loss_fpn.item()
            self.optimizer_fpn.zero_grad()
            loss_fpn.backward()
            self.optimizer_fpn.step()

            pbar.set_description(
                "Meta_train={}, Meta_test={}, Meta-test-loss = {}, Meta_train_loss={}".format(str(meta_train_list),
                                                                                              str(meta_test_list),
                                                                                              loss_meta_test,
                                                                                              loss_meta_train))
            if iter_num % (val_freq * iter_per_epoch) == 0:
                logging.info("Validation at epoch {}".format(epoch))
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                with torch.no_grad():
                    val_output = self.validate(epoch, tgt_valid_dataloader)

                if val_output['MIN_HTER'] < self.val_metrcis['MIN_HTER']:
                    self.counter = 0
                    self.val_metrcis['MIN_HTER'] = val_output['MIN_HTER']
                    self.val_metrcis['AUC'] = val_output['AUC']
                    logging.info("Save best models")
                    self.save_checkpoint(
                        {'epoch': epoch,
                         'val_metrics': self.val_metrcis,
                         'global_step': self.global_step,
                         'model_state': [self.pattern_extractor.state_dict(), self.hfn.state_dict()],
                         'optim_state': self.optimizer_fpn.state_dict(),
                         }
                    )


                logging.info('Current Best MIN_HTER={}%, AUC={}%'.format(100 * self.val_metrcis['MIN_HTER'],
                                                                         100 * self.val_metrcis['AUC']))

            # self.pattern_extractor.train()
            # self.hfn.train()
        return self.pattern_extractor

    def sync_training(self):
        """

        :return:
        """
        logging.info("Sycn training from scratch")

        self.hfn = HierachicalFusionNetwork().cuda()
        if self.config.TRAIN.IMAGENET_PRETRAIN:
            logging.info("Loading ImageNet Pretrain")
            imagenet_pretrain = torch.load('models/HFN_MP/hfn_pretrain.pth')
            self.hfn.load_state_dict(imagenet_pretrain)

        # Get hyperparameters
        init_lr = self.config.TRAIN.INIT_LR
        max_iter = self.config.TRAIN.MAX_ITER
        val_freq = self.config.TRAIN.VAL_FREQ
        metatrainsize = self.config.TRAIN.META_TRAIN_SIZE  # 2
        # Get network

        self.hfn.train()
        self.pattern_extractor.train()
        self.pattern_extractor.cuda()

        # criterionCls = nn.CrossEntropyLoss()
        criterionDepth = torch.nn.MSELoss()
        criterionCLS = torch.nn.CrossEntropyLoss()

        if self.config.TRAIN.OPTIM == 'Adam':
            self.optimizer_fpn = optim.Adam(
                chain(self.hfn.parameters(), self.pattern_extractor.parameters()),
                lr=init_lr,
                # betas=betas
            )

        elif self.config.TRAIN.OPTIM == 'SGD':
            self.optimizer_fpn = optim.SGD(
                chain(self.hfn.parameters(), self.pattern_extractor.parameters()),
                lr=self.init_lr,
                momentum=0.9
            )


        else:
            raise NotImplementedError

        # tensorboard_dir = os.path.join(self.config.OUTPUT_DIR, "tensorboard")
        # os.makedirs(tensorboard_dir, exist_ok=True)
        # self.tensorboard = None if self.config.DEBUG else SummaryWriter(tensorboard_dir)

        src1_train_dataloader_fake, src1_train_dataloader_real, \
        src2_train_dataloader_fake, src2_train_dataloader_real, \
        src3_train_dataloader_fake, src3_train_dataloader_real, \
        tgt_valid_dataloader = self.get_dataloader()

        epoch = 1

        iter_per_epoch = self.config.TRAIN.ITER_PER_EPOCH
        src1_train_dataloader_real = {
            "data_loader": src1_train_dataloader_real,
            "iterator": iter(src1_train_dataloader_real)
        }
        src2_train_dataloader_real = {
            "data_loader": src2_train_dataloader_real,
            "iterator": iter(src2_train_dataloader_real)
        }
        src3_train_dataloader_real = {
            "data_loader": src3_train_dataloader_real,
            "iterator": iter(src3_train_dataloader_real)
        }

        train_loader_list_real = [src1_train_dataloader_real, src2_train_dataloader_real, src3_train_dataloader_real]

        src1_train_dataloader_fake = {
            "data_loader": src1_train_dataloader_fake,
            "iterator": iter(src1_train_dataloader_fake)
        }
        src2_train_dataloader_fake = {
            "data_loader": src2_train_dataloader_fake,
            "iterator": iter(src2_train_dataloader_fake)
        }
        src3_train_dataloader_fake = {
            "data_loader": src3_train_dataloader_fake,
            "iterator": iter(src3_train_dataloader_fake)
        }

        train_loader_list_fake = [src1_train_dataloader_fake, src2_train_dataloader_fake, src3_train_dataloader_fake]

        pbar = tqdm(range(max_iter + 1), ncols=160)
        self.pattern_extractor.train()
        self.hfn.train()
        for iter_num in pbar:
            if (iter_num != 0 and iter_num % iter_per_epoch == 0):
                epoch = epoch + 1
                logging.info("Training at Epoch {}".format(epoch))

            # Load Meta-train data
            image_meta_train, label_meta_train, depth_meta_train = get_data_from_loader_list(
                train_loader_list_real,
                train_loader_list_fake)

            image_meta_train = image_meta_train.cuda()
            depth_meta_train = [d.cuda() for d in depth_meta_train] #.cuda()

            # Target Network does the inference with image_meta_train
            img_colored, _ = self.pattern_extractor(image_meta_train,)

            # Calculate meta-train loss
            depth_pred, cls_preds = self.hfn(image_meta_train, img_colored)  # TODO

            mse_loss = criterionDepth(depth_pred[0].squeeze(), depth_meta_train[0])
            cls_loss = criterionCLS(cls_preds[0], label_meta_train.cuda())

            loss_fpn = mse_loss + cls_loss
            self.optimizer_fpn.zero_grad()
            loss_fpn.backward()
            self.optimizer_fpn.step()
            pbar.set_description('MSE_LOSS={:.4f}, CLS_LOSS={:.4f}'.format(mse_loss.item(), cls_loss.item()))
            # For update fpn

            if iter_num % (val_freq * iter_per_epoch) == 0:
                logging.info("Validation at epoch {}".format(epoch))

                with torch.no_grad():
                    val_output = self.validate(epoch, tgt_valid_dataloader)
                self.pattern_extractor.train()
                self.hfn.train()
                if val_output['MIN_HTER'] < self.val_metrcis['MIN_HTER']:
                    self.counter = 0
                    self.val_metrcis['MIN_HTER'] = val_output['MIN_HTER']
                    self.val_metrcis['AUC'] = val_output['AUC']
                    self.save_checkpoint(
                        {'epoch': epoch,
                         'val_metrics': self.val_metrcis,
                         'global_step': self.global_step,
                         'model_state': [self.pattern_extractor.state_dict(), self.hfn.state_dict()],

                         'optim_state': self.optimizer_fpn.state_dict(),
                         }
                    )

                else:
                    self.counter += 1

                logging.info('Current Best MIN_HTER={}%, AUC={}%'.format(100 * self.val_metrcis['MIN_HTER'],
                                                                         100 * self.val_metrcis['AUC']))
                if self.counter > self.train_patience:
                    logging.info("[!] No improvement in a while, stopping training.")
                    self.save_checkpoint(
                        {'epoch': epoch,
                         'val_metrics': self.val_metrcis,
                         'global_step': self.global_step,
                         'model_state': [self.pattern_extractor.state_dict(), self.hfn.state_dict(),
                                         ],
                         'optim_state': self.optimizer_fpn.state_dict(),
                         }
                    )



    def train_hfn_from_scratch(self, pattern_extractor=None):
        """

        :return:
        """
        if pattern_extractor is not None:
            self.pattern_extractor = pattern_extractor
        logging.info("train_hfn_from_scratch")
        ckpt_path = os.path.join(self.config.OUTPUT_DIR, 'ckpt/best.ckpt')
        if self.config.TRAIN.RESUME:
            ckpt_path = self.config.TRAIN.RESUME

        state_dict = torch.load(ckpt_path)
        self.hfn = HierachicalFusionNetwork().cuda()
        if self.config.TRAIN.IMAGENET_PRETRAIN:
            # logging.info("Loading ImageNet Pretrain")
            imagenet_pretrain = torch.load('models/HFN_MP/hfn_pretrain.pth')
            self.hfn.load_state_dict(imagenet_pretrain)


        self.pattern_extractor.cuda()
        self.pattern_extractor.load_state_dict(state_dict['model_state'][0])
        self.pattern_extractor.eval()

        # Get hyperparameters
        init_lr = self.config.TRAIN.INIT_LR
        max_iter = self.config.TRAIN.MAX_ITER
        val_freq = self.config.TRAIN.VAL_FREQ
        metatrainsize = self.config.TRAIN.META_TRAIN_SIZE  # 2
        # Get network

        for param in self.pattern_extractor.parameters():
            param.requires_grad = False

        self.hfn.train()

        # criterionCls = nn.CrossEntropyLoss()
        criterionDepth = torch.nn.MSELoss()
        criterionCLS = torch.nn.CrossEntropyLoss()

        if self.config.TRAIN.OPTIM == 'Adam':
            self.optimizer_fpn = optim.Adam(
                self.hfn.parameters(),
                lr=init_lr,
                # betas=betas
            )

        elif self.config.TRAIN.OPTIM == 'SGD':
            self.optimizer_fpn = optim.SGD(
                self.hfn.parameters(),
                lr=self.init_lr,
                momentum=0.9
            )


        else:
            raise NotImplementedError

        # tensorboard_dir = os.path.join(self.config.OUTPUT_DIR, "tensorboard")
        # os.makedirs(tensorboard_dir, exist_ok=True)
        # self.tensorboard = None if self.config.DEBUG else SummaryWriter(tensorboard_dir)

        src1_train_dataloader_fake, src1_train_dataloader_real, \
        src2_train_dataloader_fake, src2_train_dataloader_real, \
        src3_train_dataloader_fake, src3_train_dataloader_real, \
        tgt_valid_dataloader = self.get_dataloader()

        epoch = 1

        iter_per_epoch = self.config.TRAIN.ITER_PER_EPOCH
        src1_train_dataloader_real = {
            "data_loader": src1_train_dataloader_real,
            "iterator": iter(src1_train_dataloader_real)
        }
        src2_train_dataloader_real = {
            "data_loader": src2_train_dataloader_real,
            "iterator": iter(src2_train_dataloader_real)
        }
        src3_train_dataloader_real = {
            "data_loader": src3_train_dataloader_real,
            "iterator": iter(src3_train_dataloader_real)
        }

        train_loader_list_real = [src1_train_dataloader_real, src2_train_dataloader_real, src3_train_dataloader_real]

        src1_train_dataloader_fake = {
            "data_loader": src1_train_dataloader_fake,
            "iterator": iter(src1_train_dataloader_fake)
        }
        src2_train_dataloader_fake = {
            "data_loader": src2_train_dataloader_fake,
            "iterator": iter(src2_train_dataloader_fake)
        }
        src3_train_dataloader_fake = {
            "data_loader": src3_train_dataloader_fake,
            "iterator": iter(src3_train_dataloader_fake)
        }

        train_loader_list_fake = [src1_train_dataloader_fake, src2_train_dataloader_fake, src3_train_dataloader_fake]

        pbar = tqdm(range(max_iter + 1), ncols=160)

        for iter_num in pbar:
            if (iter_num != 0 and iter_num % iter_per_epoch == 0):
                epoch = epoch + 1
                logging.info("Training at Epoch {}".format(epoch))

            # Load Meta-train data
            image_meta_train, label_meta_train, depth_meta_train = get_data_from_loader_list(
                train_loader_list_real,
                train_loader_list_fake)

            image_meta_train = image_meta_train.cuda()
            depth_meta_train = [d.cuda() for d in depth_meta_train] #.cuda()
            # Target Network does the inference with image_meta_train
            img_colored, _ = self.pattern_extractor(image_meta_train, )


            # Calculate meta-train loss
            depth_pred, cls_preds = self.hfn(image_meta_train, img_colored)  # TODO

            mse_loss = criterionDepth(depth_pred[0].squeeze(), depth_meta_train[0])
            cls_loss = criterionCLS(cls_preds[0], label_meta_train.cuda())

            loss_fpn = mse_loss + cls_loss
            self.optimizer_fpn.zero_grad()
            loss_fpn.backward()
            self.optimizer_fpn.step()
            pbar.set_description('MSE_LOSS={:.4f}, CLS_LOSS={:.4f}'.format(mse_loss.item(), cls_loss.item()))
            # For update fpn

            if iter_num % (val_freq * iter_per_epoch) == 0:
                logging.info("Validation at epoch {}".format(epoch))

                with torch.no_grad():
                    val_output = self.validate(epoch, tgt_valid_dataloader)

                if val_output['MIN_HTER'] < self.val_metrcis['MIN_HTER']:
                    self.counter = 0
                    self.val_metrcis['MIN_HTER'] = val_output['MIN_HTER']
                    self.val_metrcis['AUC'] = val_output['AUC']
                    self.save_checkpoint(
                        {'epoch': epoch,
                         'val_metrics': self.val_metrcis,
                         'global_step': self.global_step,
                         'model_state': [self.pattern_extractor.state_dict(), self.hfn.state_dict()],

                         'optim_state': self.optimizer_fpn.state_dict(),
                         }
                    )

                else:
                    self.counter += 1

                logging.info('Current Best MIN_HTER={}%, AUC={}%'.format(100 * self.val_metrcis['MIN_HTER'],
                                                                         100 * self.val_metrcis['AUC']))
                if self.counter > self.train_patience:
                    logging.info("[!] No improvement in a while, stopping training.")
                    self.save_checkpoint(
                        {'epoch': epoch,
                         'val_metrics': self.val_metrcis,
                         'global_step': self.global_step,
                         'model_state': [self.pattern_extractor.state_dict(), self.hfn.state_dict(),
                                         ],
                         'optim_state': self.optimizer_fpn.state_dict(),
                         }
                    )

            self.pattern_extractor.train()
            self.hfn.train()
            return self.hfn

    def test(self, test_data_loader):

        criterionDepth = torch.nn.MSELoss()
        criterionCLS = torch.nn.CrossEntropyLoss()
        avg_test_loss = AverageMeter()

        scores_pred_dict = {}
        face_label_gt_dict = {}
        map_scores_pred_dict = {}
        tmp_dict1 = {}
        bc_scores_pred_dict = {}
        tmp_dict2 = {}

        self.pattern_extractor.eval()
        self.hfn.eval()
        with torch.no_grad():
            for data in tqdm(test_data_loader, ncols=80):
                network_input, target, video_ids = data[1], data[2], data[3]

                map_targets, face_targets = target['pixel_maps'], target['face_label'].cuda()

                map_targets = [x.cuda() for x in map_targets] #.cuda()

                network_input = network_input.cuda()
                img_colored, _ = self.pattern_extractor(network_input)

                depth_pred, cls_preds = self.hfn(network_input, img_colored)  # TODO

                mse_loss = criterionDepth(depth_pred[0].squeeze(), map_targets[0])
                cls_loss = criterionCLS(cls_preds[0], face_targets)

                map_score = depth_pred[0].squeeze(1).mean(dim=[1, 2])
                cls_score = torch.softmax(cls_preds[0], 1)[:, 1]
                score = map_score + cls_score
                score /= 2

                mse_loss += criterionDepth(depth_pred[0].squeeze(), map_targets[0])

                test_loss = (mse_loss + cls_loss) / 2
                avg_test_loss.update(test_loss.item(), network_input.size()[0])

                pred_score = score.cpu().numpy()
                gt_dict, pred_dict = self._collect_scores_from_loader(scores_pred_dict, face_label_gt_dict,
                                                                      target['face_label'].numpy(), pred_score,
                                                                      video_ids
                                                                      )
                tmp_dict, map_pred_dict = self._collect_scores_from_loader(map_scores_pred_dict, tmp_dict1,
                                                                      target['face_label'].numpy(), map_score.cpu().numpy(),
                                                                      video_ids
                                                                      )

                tmp_dict, bc_pred_dict = self._collect_scores_from_loader(bc_scores_pred_dict, tmp_dict2,
                                                                      target['face_label'].numpy(), cls_score.cpu().numpy(),
                                                                      video_ids
                                                                      )

        test_results = {
            'scores_gt': gt_dict,
            'scores_pred': pred_dict,
            'avg_loss': avg_test_loss.avg,
            'map_scores': map_pred_dict,
            'cls_scores': bc_pred_dict

        }
        return test_results


def get_data_from_pair_loaders(real_loader, fake_loader):
    # try:
    #     _, img_real, target_real, _ = real_loader['iterator'].next()
    # except:
    #     real_loader['iterator'] = iter(real_loader['data_loader'])
    #     _, img_real, target_real, _ = real_loader['iterator'].next()
    try:
    # Use the built-in `next()` function
        _, img_real, target_real, _ = next(real_loader['iterator'])
    except StopIteration:
    # Reset the iterator if StopIteration is raised
        real_loader['iterator'] = iter(real_loader['data_loader'])
        _, img_real, target_real, _ = next(real_loader['iterator'])
    label_real = target_real['face_label'].cuda()
    pixel_maps_real = target_real['pixel_maps']

    # try:
    #     _, img_fake, target_fake, _ = fake_loader['iterator'].next()
    # except:
    #     fake_loader['iterator'] = iter(fake_loader['data_loader'])
    #     _, img_fake, target_fake, _ = fake_loader['iterator'].next()
    try:
        # Use Python's built-in next() function
        _, img_fake, target_fake, _ = next(fake_loader['iterator'])
    except StopIteration:
        # Reset the iterator if StopIteration is raised
        fake_loader['iterator'] = iter(fake_loader['data_loader'])
        _, img_fake, target_fake, _ = next(fake_loader['iterator'])

    label_fake = target_fake['face_label'].cuda()
    pixel_maps_fake = target_fake['pixel_maps']

    img = torch.cat([img_real, img_fake], dim=0)
    label = torch.cat([label_real, label_fake], dim=0)
    pixel_maps = [torch.cat([pixel_maps_real[i], pixel_maps_fake[i]], 0) for i in range(5)]

    return img, label, pixel_maps


def get_data_from_loader_list(real_loader_list, fake_loader_list):
    img_list = []
    label_list = []
    pixel_maps_list = []
    if len(real_loader_list) == 1:
        return get_data_from_pair_loaders(real_loader_list[0], fake_loader_list[0])

    else:

        for real_loader, fake_loader in zip(real_loader_list, fake_loader_list):
            img, label, pixel_maps = get_data_from_pair_loaders(real_loader, fake_loader)
            img_list.append(img)
            label_list.append(label)
            pixel_maps_list.append(pixel_maps)

        imgs = torch.cat(img_list, 0)
        labels = torch.cat(label_list, 0)

        pixel_maps = [torch.cat(maps, 0) for maps in zip(*pixel_maps_list)]

        return imgs, labels, pixel_maps


def get_meta_train_and_test_data(real_loaders, fake_loaders):
    num_src_domains = len(real_loaders)
    domain_list = list(range(num_src_domains))

    random.shuffle(domain_list)

    meta_train_list = domain_list[2:]
    meta_test_list = domain_list[:1]

    meta_train_loader_list_real = [real_loaders[i] for i in meta_train_list]
    meta_train_loader_list_fake = [fake_loaders[i] for i in meta_train_list]

    meta_test_loader_list_real = [real_loaders[i] for i in meta_test_list]
    meta_test_loader_list_fake = [fake_loaders[i] for i in meta_test_list]

    imgs_train, labels_train, pixel_maps_train = get_data_from_loader_list(meta_train_loader_list_real,
                                                                           meta_train_loader_list_fake)
    imgs_test, labels_test, pixel_maps_test = get_data_from_loader_list(meta_test_loader_list_real,
                                                                        meta_test_loader_list_fake)
    return (imgs_train, labels_train, pixel_maps_train), (imgs_test, labels_test, pixel_maps_test)


def zero_param_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()
