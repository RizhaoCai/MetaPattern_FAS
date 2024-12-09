import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import logging
from torch.optim import SGD
from typing import Dict, List, Tuple, Optional
from copy import deepcopy

from .trainer import Trainer, get_data_from_loader_list
from .meta_network import MetaHierachicalFusionNetwork, MetaPatternExtractor

class BaseMetaTrainer(Trainer): # 
    """Base class for meta-learning trainers"""
    def __init__(self, config):
        super(BaseMetaTrainer, self).__init__(config)
        self.pattern_extractor = MetaPatternExtractor().cuda()
        # self.hfn = MetaHierachicalFusionNetwork(
        #     mean_std_normalize=config.MODEL.MEAN_STD_NORMAL,
        #     dropout_rate=config.TRAIN.DROPOUT
        # ).cuda()
        
        self.train_metrics = {
            "loss": [],
            "accuracy": [],
            "meta_loss": []
        }
        self._setup_training()
        self.tensorboard = None

    def _setup_training(self):
        self.criterionDepth = nn.MSELoss()
        self.criterionReconstruction = nn.MSELoss()
        self.criterionCLS = nn.CrossEntropyLoss()
        self.inner_lr = self.config.TRAIN.INNER_LR
        self.meta_lr = self.config.TRAIN.META_LEARNING_RATE
        self.inner_steps = self.config.TRAIN.INNER_LOOPS
        self.real_loaders, self.fake_loaders, self.tgt_valid_dataloader = self._get_data_loaders()

    def _get_data_loaders(self):
        src1_train_dataloader_fake, src1_train_dataloader_real, \
        src2_train_dataloader_fake, src2_train_dataloader_real, \
        src3_train_dataloader_fake, src3_train_dataloader_real, \
        tgt_valid_dataloader = self.get_dataloader()

        real_loaders = [
            {"data_loader": src1_train_dataloader_real, "iterator": iter(src1_train_dataloader_real)},
            {"data_loader": src2_train_dataloader_real, "iterator": iter(src2_train_dataloader_real)},
            {"data_loader": src3_train_dataloader_real, "iterator": iter(src3_train_dataloader_real)}
        ]
        fake_loaders = [
            {"data_loader": src1_train_dataloader_fake, "iterator": iter(src1_train_dataloader_fake)},
            {"data_loader": src2_train_dataloader_fake, "iterator": iter(src2_train_dataloader_fake)},
            {"data_loader": src3_train_dataloader_fake, "iterator": iter(src3_train_dataloader_fake)}
        ]

        return real_loaders, fake_loaders, tgt_valid_dataloader

    def _compute_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        img_colored, reconstruct_rgb = outputs['colored']
        depth_pred, cls_preds = outputs['predictions']
        
        reconstruction_loss = (
            self.criterionReconstruction(reconstruct_rgb, targets['image'])
            if reconstruct_rgb is not None else 0
        )
        
        depth_loss = self.criterionDepth(
            depth_pred[0].squeeze(),
            targets['depth'][0]
        )
        
        cls_loss = self.criterionCLS(
            cls_preds[0],
            targets['label']
        )
        
        return depth_loss + cls_loss + reconstruction_loss

    def _compute_accuracy(self, outputs, targets):
        _, cls_preds = outputs['predictions']
        _, preds = torch.max(cls_preds[0], dim=1)
        correct = (preds == targets['label']).sum().item()
        total = targets['label'].size(0)
        return correct / total

    def _forward_pass(self, 
                    image: torch.Tensor,
                    extractor_params: Optional[OrderedDict] = None,
                    hfn_params: Optional[OrderedDict] = None) -> Dict[str, torch.Tensor]:
        if not image.is_cuda:
            image = image.cuda()
            
        img_colored, reconstruct_rgb = self.pattern_extractor(
            image,
            params=extractor_params
        )

        if hfn_params is not None:
            depth_pred, cls_preds = self.hfn(image, img_colored, params=hfn_params)
        else:
            depth_pred, cls_preds = self.hfn(image, img_colored)
        
        return {
            'colored': (img_colored, reconstruct_rgb),
            'predictions': (depth_pred, cls_preds)
        }

    def _validate_and_save(self, iter_num: int):
        logging.info(f"Validation at iteration {iter_num}")
        with torch.no_grad():
            val_output = self.validate(iter_num, self.tgt_valid_dataloader)
            
            if val_output['MIN_HTER'] < self.val_metrcis['MIN_HTER']:
                self.val_metrcis.update(val_output)
                logging.info("Save best models")
                self.save_checkpoint({
                    'epoch': iter_num // self.config.TRAIN.ITER_PER_EPOCH,
                    'val_metrics': self.val_metrcis,
                    'model_state': [
                        self.pattern_extractor.state_dict(),
                        self.hfn.state_dict()
                    ],
                    'optim_state': self.meta_optimizer.state_dict(),
                })

    def train_hfn_from_scratch(self):
        """Train HFN from scratch using adapted pattern extractor."""
        logging.info("Training HFN from scratch")
        
        for param in self.pattern_extractor.parameters():
            param.requires_grad = False
            
        hfn_optimizer = optim.Adam(
            self.hfn.parameters(),
            lr=self.config.TRAIN.INIT_LR
        )
        
        pbar = tqdm(range(1, self.config.TRAIN.MAX_ITER + 1), ncols=160)
        epoch_loss = 0.0
        epoch_acc = 0.0
        batch_count = 0
        
        for iter_num in pbar:
            batch_data = get_data_from_loader_list(self.real_loaders, self.fake_loaders)
            
            images = batch_data[0].cuda()
            labels = batch_data[1].cuda()
            depths = [d.cuda() for d in batch_data[2]]
            
            outputs = self._forward_pass(images)
            targets = {
                'image': images,
                'label': labels,
                'depth': depths
            }
            
            loss = self._compute_loss(outputs, targets)
            acc = self._compute_accuracy(outputs, targets)
            
            hfn_optimizer.zero_grad()
            loss.backward()
            hfn_optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc
            batch_count += 1
            
            if iter_num % self.config.TRAIN.ITER_PER_EPOCH == 0:
                avg_loss = epoch_loss / batch_count
                avg_acc = epoch_acc / batch_count
                logging.info(f"Iteration {iter_num}: Train Loss = {avg_loss:.4f}, Train Accuracy = {avg_acc:.4f}")
                epoch_loss = 0.0
                epoch_acc = 0.0
                batch_count = 0
            
            if iter_num % (self.config.TRAIN.VAL_FREQ * self.config.TRAIN.ITER_PER_EPOCH) == 0:
                self._validate_and_save(iter_num)

        for param in self.pattern_extractor.parameters():
            param.requires_grad = True


class PatternExtractorMAMLTrainer(BaseMetaTrainer):
    def __init__(self, config, hfn):
        super(PatternExtractorMAMLTrainer, self).__init__(config)
        self.meta_optimizer = optim.Adam(
            self.pattern_extractor.parameters(),
            lr=self.meta_lr
        )
        self.hfn = hfn
        # self.hfn.eval()
    def _maml_adaptation(self, support_data: Tuple[torch.Tensor, ...]) -> Tuple[OrderedDict, float]:
        images, labels, depths = support_data
        images = images.cuda()
        labels = labels.cuda()
        depths = [d.cuda() for d in depths]

        adapted_extractor_dict = OrderedDict(
            (name, param.clone())
            for name, param in self.pattern_extractor.named_parameters()
            if param.requires_grad
        )

        for _ in range(self.inner_steps):
            img_colored, reconstruct_rgb = self.pattern_extractor(images, params=adapted_extractor_dict)
            depth_pred, cls_preds = self.hfn(images, img_colored)
            
            outputs = {
                'colored': (img_colored, reconstruct_rgb),
                'predictions': (depth_pred, cls_preds)
            }
            
            loss = self._compute_loss(outputs, {'image': images, 'label': labels, 'depth': depths})
            
            grads = torch.autograd.grad(
                loss, 
                adapted_extractor_dict.values(),
                create_graph=True,
                allow_unused=True
            )
            
            for (name, param), grad in zip(adapted_extractor_dict.items(), grads):
                if grad is not None:
                    adapted_extractor_dict[name] = param - self.inner_lr * grad

        return adapted_extractor_dict, loss.item()

    def train(self):
        logging.info("Starting MAML Feature Extractor Training")
        pattern_extractor = self._maml_train()
        return pattern_extractor


    def _maml_train(self):
        pbar = tqdm(range(1, self.config.TRAIN.MAX_ITER + 1), ncols=160)
        epoch_loss = 0.0
        epoch_acc = 0.0
        batch_count = 0
        
        for iter_num in pbar:
            meta_loss = 0.0
            batch_acc = 0.0
            self.meta_optimizer.zero_grad()
            
            domain_list = list(range(3))
            random.shuffle(domain_list)
            support_list = domain_list[:self.config.TRAIN.META_TRAIN_SIZE]
            query_list = domain_list[self.config.TRAIN.META_TRAIN_SIZE:]
            for task_idx in support_list:
                support_data = get_data_from_loader_list(
                    [self.real_loaders[task_idx]],
                    [self.fake_loaders[task_idx]]
                )
                
                adapted_params, support_loss = self._maml_adaptation(support_data)
            for task_idx in query_list:   
                query_data = get_data_from_loader_list(
                    [self.real_loaders[task_idx]],
                    [self.fake_loaders[task_idx]]
                )
                
                query_images = query_data[0].cuda()
                query_labels = query_data[1].cuda()
                query_depths = [d.cuda() for d in query_data[2]]
                
                query_outputs = self._forward_pass(query_images, adapted_params)
                query_targets = {
                    'image': query_images,
                    'label': query_labels,
                    'depth': query_depths
                }
                
                query_loss = self._compute_loss(query_outputs, query_targets)
                batch_acc += self._compute_accuracy(query_outputs, query_targets)
                meta_loss += query_loss
            
            meta_loss = meta_loss / len(query_list)
            batch_acc = batch_acc / len(query_list)
            meta_loss.backward()
            self.meta_optimizer.step()
            
            epoch_loss += meta_loss.item()
            epoch_acc += batch_acc
            batch_count += 1
            
            if iter_num % self.config.TRAIN.ITER_PER_EPOCH == 0:
                self._log_training_stats(iter_num, epoch_loss, epoch_acc, batch_count)
                epoch_loss = 0.0
                epoch_acc = 0.0
                batch_count = 0
            
            if iter_num % (self.config.TRAIN.VAL_FREQ * self.config.TRAIN.ITER_PER_EPOCH) == 0:
                self._validate_and_save(iter_num)
        return self.pattern_extractor
    def _log_training_stats(self, iter_num, epoch_loss, epoch_acc, batch_count):
        avg_loss = epoch_loss / batch_count
        avg_acc = epoch_acc / batch_count
        logging.info(f"Iteration {iter_num}: Train Loss = {avg_loss:.4f}, Train Accuracy = {avg_acc:.4f}")


class PatternExtractorReptileTrainer(BaseMetaTrainer):
    def __init__(self, config, hfn):
        super(PatternExtractorReptileTrainer, self).__init__(config)
        self.meta_optimizer = optim.Adam(
            self.pattern_extractor.parameters(),
            lr=self.meta_lr
        )
        self.hfn = hfn
        # self.hfn.eval()
    def _reptile_inner_loop(self, task_data: Tuple[torch.Tensor, ...], num_steps: int) -> Tuple[OrderedDict, OrderedDict]:
       images, labels, depths = task_data
       images = images.cuda()
       labels = labels.cuda()
       depths = [d.cuda() for d in depths]
       
       initial_extractor_params = OrderedDict(
           (name, param.clone()) 
           for name, param in self.pattern_extractor.named_parameters()
       )
       initial_hfn_params = OrderedDict(
           (name, param.clone())
           for name, param in self.hfn.named_parameters() 
       )
       
       inner_optimizer_extractor = SGD(self.pattern_extractor.parameters(), lr=self.inner_lr)
       inner_optimizer_hfn = SGD(self.hfn.parameters(), lr=self.inner_lr)
       
       for _ in range(num_steps):
           img_colored, reconstruct_rgb = self.pattern_extractor(images)
           depth_pred, cls_preds = self.hfn(images, img_colored)
           
           outputs = {
               'colored': (img_colored, reconstruct_rgb),
               'predictions': (depth_pred, cls_preds)
           }
           
           loss = self._compute_loss(outputs, {'image': images, 'label': labels, 'depth': depths})
           
           inner_optimizer_extractor.zero_grad()
           inner_optimizer_hfn.zero_grad()
           loss.backward()
           inner_optimizer_extractor.step()
           inner_optimizer_hfn.step()
       
       final_extractor_params = OrderedDict(
           (name, param.clone())
           for name, param in self.pattern_extractor.named_parameters()
       )
       final_hfn_params = OrderedDict(
           (name, param.clone())
           for name, param in self.hfn.named_parameters()
       )
       
       for name, param in self.pattern_extractor.named_parameters():
           param.data.copy_(initial_extractor_params[name].data)
       for name, param in self.hfn.named_parameters():
           param.data.copy_(initial_hfn_params[name].data)
       
       return final_extractor_params, final_hfn_params

    def train(self):
        logging.info("Starting Reptile Feature Extractor Training")
        pattern_extractor = self._reptile_train()
        return pattern_extractor

    def _reptile_train(self):
        pbar = tqdm(range(1, self.config.TRAIN.MAX_ITER + 1), ncols=160)
        epoch_loss = 0.0
        epoch_acc = 0.0
        batch_count = 0
        
        for iter_num in pbar:
            task_idx = random.randint(0, 2)
            task_data = get_data_from_loader_list(
                [self.real_loaders[task_idx]],
                [self.fake_loaders[task_idx]]
            )
            
            final_extractor_params, final_hfn_params = self._reptile_inner_loop(task_data, self.inner_steps)
            meta_lr = self.meta_lr * (1 - iter_num / self.config.TRAIN.MAX_ITER)
            for name, param in self.pattern_extractor.named_parameters():
                param.data.add_(meta_lr * (final_extractor_params[name].data - param.data))
            for name, param in self.hfn.named_parameters():
                param.data.add_(meta_lr * (final_hfn_params[name].data - param.data))

            images = task_data[0].cuda()
            labels = task_data[1].cuda()
            depths = [d.cuda() for d in task_data[2]]
            
            outputs = self._forward_pass(images)
            targets = {
                'image': images,
                'label': labels,
                'depth': depths
            }
            
            current_loss = self._compute_loss(outputs, targets)
            current_acc = self._compute_accuracy(outputs, targets)
            
            epoch_loss += current_loss.item()
            epoch_acc += current_acc
            batch_count += 1
            
            if iter_num % self.config.TRAIN.ITER_PER_EPOCH == 0:
                self._log_training_stats(iter_num, epoch_loss, epoch_acc, batch_count)
                epoch_loss = 0.0
                epoch_acc = 0.0
                batch_count = 0
            
            if iter_num % (self.config.TRAIN.VAL_FREQ * self.config.TRAIN.ITER_PER_EPOCH) == 0:
                self._validate_and_save(iter_num)
        return self.pattern_extractor
    
    def _log_training_stats(self, iter_num, epoch_loss, epoch_acc, batch_count):
        avg_loss = epoch_loss / batch_count
        avg_acc = epoch_acc / batch_count
        logging.info(f"Iteration {iter_num}: Train Loss = {avg_loss:.4f}, Train Accuracy = {avg_acc:.4f}")

class HFNMAMLTrainer(BaseMetaTrainer):
    def __init__(self, config, pattern_extractor):
        super(HFNMAMLTrainer, self).__init__(config)
        self.meta_optimizer = optim.Adam(
            self.hfn.parameters(),
            lr=self.meta_lr
        )
        if pattern_extractor is None:
            raise ValueError("Pattern extractor cannot be None in HFNMAMLTrainer.")
        self.pattern_extractor = pattern_extractor
        self.hfn = MetaHierachicalFusionNetwork(
            mean_std_normalize=config.MODEL.MEAN_STD_NORMAL,
            dropout_rate=config.TRAIN.DROPOUT
        ).cuda()
    def _maml_adaptation(self, support_data: Tuple[torch.Tensor, ...]) -> Tuple[OrderedDict, float]:
        images, labels, depths = support_data
        images = images.cuda()
        labels = labels.cuda()
        depths = [d.cuda() for d in depths]
        
        adapted_hfn_params = OrderedDict(
            (name, param.clone())
            for name, param in self.hfn.named_parameters()
            if param.requires_grad
        )
        
        for _ in range(self.inner_steps):
            with torch.no_grad():
                img_colored, _ = self.pattern_extractor(images)
            
            depth_pred, cls_preds = self.hfn(images, img_colored, params=adapted_hfn_params)
            
            outputs = {
                'colored': (None, None),
                'predictions': (depth_pred, cls_preds)
            }
            loss = self._compute_loss(outputs, {'image': images, 'label': labels, 'depth': depths})
            
            grads = torch.autograd.grad(
                loss,
                adapted_hfn_params.values(),
                create_graph=True,
                allow_unused=True
            )
            
            for (name, param), grad in zip(adapted_hfn_params.items(), grads):
                if grad is not None:
                    adapted_hfn_params[name] = param - self.inner_lr * grad

        return adapted_hfn_params, loss.item()
    
    def train(self):
        logging.info("Starting MAML HFN Training")
        self._maml_train()
        return self.hfn

    def _maml_train(self):
        pbar = tqdm(range(1, self.config.TRAIN.MAX_ITER + 1), ncols=160)
        epoch_loss = 0.0
        batch_count = 0
        
        for iter_num in pbar:
            meta_loss = 0.0
            self.meta_optimizer.zero_grad()
            
            domain_list = list(range(3))
            random.shuffle(domain_list)
            support_list = domain_list[:self.config.TRAIN.META_TRAIN_SIZE]
            query_list = domain_list[self.config.TRAIN.META_TRAIN_SIZE:]
            for task_idx in support_list:
                support_data = get_data_from_loader_list(
                    [self.real_loaders[task_idx]],
                    [self.fake_loaders[task_idx]]
                )
                
                adapted_hfn_params, support_loss = self._maml_adaptation(support_data)
            
            for task_idx in query_list:
                query_data = get_data_from_loader_list(
                    [self.real_loaders[task_idx]],
                    [self.fake_loaders[task_idx]]
                )
                query_images = query_data[0].cuda()
                query_labels = query_data[1].cuda()
                query_depths = [d.cuda() for d in query_data[2]]
                
                with torch.no_grad():
                    img_colored, _ = self.pattern_extractor(query_images)
                depth_pred, cls_preds = self.hfn(query_images, img_colored, params=adapted_hfn_params)
                
                outputs = {
                    'colored': (None, None),
                    'predictions': (depth_pred, cls_preds)
                }
                query_loss = self._compute_loss(outputs, {'image': query_images, 'label': query_labels, 'depth': query_depths})
                meta_loss += query_loss
            
            meta_loss = meta_loss / len(query_list)
            meta_loss.backward()
            self.meta_optimizer.step()
            
            epoch_loss += meta_loss.item()
            batch_count += 1
            
            if iter_num % self.config.TRAIN.ITER_PER_EPOCH == 0:
                avg_loss = epoch_loss / batch_count
                logging.info(f"Iteration {iter_num}: Meta-Train Loss = {avg_loss:.4f}")
                epoch_loss = 0.0
                batch_count = 0
            
            if iter_num % (self.config.TRAIN.VAL_FREQ * self.config.TRAIN.ITER_PER_EPOCH) == 0:
                self._validate_and_save(iter_num)


class HFNReptileTrainer(BaseMetaTrainer):
    def __init__(self, config, pattern_extractor):
        super(HFNReptileTrainer, self).__init__(config)
        self.meta_optimizer = optim.Adam(
            self.hfn.parameters(),
            lr=self.meta_lr
        )
        if pattern_extractor is None:
            raise ValueError("Pattern extractor cannot be None in HFNMAMLTrainer.")
        self.pattern_extractor = pattern_extractor
    def _reptile_inner_loop(self, task_data: Tuple[torch.Tensor, ...], num_steps: int) -> OrderedDict:
        images, labels, depths = task_data
        images = images.cuda()
        labels = labels.cuda()
        depths = [d.cuda() for d in depths]
        
        initial_hfn_params = OrderedDict(
            (name, param.clone())
            for name, param in self.hfn.named_parameters()
        )
        
        optimizer = SGD(self.hfn.parameters(), lr=self.inner_lr)
        for _ in range(num_steps):
            with torch.no_grad():
                img_colored, _ = self.pattern_extractor(images)
            
            depth_pred, cls_preds = self.hfn(images, img_colored)
            
            outputs = {
                'colored': (None, None),
                'predictions': (depth_pred, cls_preds)
            }
            loss = self._compute_loss(outputs, {'image': images, 'label': labels, 'depth': depths})
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        final_hfn_params = OrderedDict(
            (name, param.clone())
            for name, param in self.hfn.named_parameters()
        )
        
        for name, param in self.hfn.named_parameters():
            param.data.copy_(initial_hfn_params[name].data)
        
        return final_hfn_params
    
    def train(self):
        logging.info("Starting Reptile HFN Training")
        self._reptile_train()
        return self.hfn
    
    def _reptile_train(self):
        pbar = tqdm(range(1, self.config.TRAIN.MAX_ITER + 1), ncols=160)
        epoch_loss = 0.0
        batch_count = 0
        
        for iter_num in pbar:
            task_idx = random.randint(0, 2)
            task_data = get_data_from_loader_list(
                [self.real_loaders[task_idx]],
                [self.fake_loaders[task_idx]]
            )
            
            final_hfn_params = self._reptile_inner_loop(task_data, self.inner_steps)
            
            meta_lr = self.meta_lr * (1 - iter_num / self.config.TRAIN.MAX_ITER)
            for name, param in self.hfn.named_parameters():
                param.data.add_(meta_lr * (final_hfn_params[name].data - param.data))
            
            with torch.no_grad():
                images = task_data[0].cuda()
                labels = task_data[1].cuda()
                depths = [d.cuda() for d in task_data[2]]
                
                img_colored, _ = self.pattern_extractor(images)
                depth_pred, cls_preds = self.hfn(images, img_colored)
                
                outputs = {
                    'colored': (None, None),
                    'predictions': (depth_pred, cls_preds)
                }
                loss = self._compute_loss(outputs, {'image': images, 'label': labels, 'depth': depths})
                epoch_loss += loss.item()
                batch_count += 1
            
            if iter_num % self.config.TRAIN.ITER_PER_EPOCH == 0:
                avg_loss = epoch_loss / batch_count
                logging.info(f"Iteration {iter_num}: Meta-Train Loss = {avg_loss:.4f}")
                epoch_loss = 0.0
                batch_count = 0
            
            if iter_num % (self.config.TRAIN.VAL_FREQ * self.config.TRAIN.ITER_PER_EPOCH) == 0:
                self._validate_and_save(iter_num)


class MetaTrainerManager(Trainer):
    def __init__(self, config):
        super(MetaTrainerManager, self).__init__(config)
        self.config = config
        self.pattern_extractor = None  # Initialize as None
        self.hfn = None  # Initialize HFN as None if needed

        self.best_val_loss = float('inf')

        self.real_loaders, self.fake_loaders, self.val_loader = self._get_data_loaders()

        
    def _get_data_loaders(self):
        src1_train_dataloader_fake, src1_train_dataloader_real, \
        src2_train_dataloader_fake, src2_train_dataloader_real, \
        src3_train_dataloader_fake, src3_train_dataloader_real, \
        tgt_valid_dataloader = self.get_dataloader()

        real_loaders = [
            {"data_loader": src1_train_dataloader_real, "iterator": iter(src1_train_dataloader_real)},
            {"data_loader": src2_train_dataloader_real, "iterator": iter(src2_train_dataloader_real)},
            {"data_loader": src3_train_dataloader_real, "iterator": iter(src3_train_dataloader_real)}
        ]
        fake_loaders = [
            {"data_loader": src1_train_dataloader_fake, "iterator": iter(src1_train_dataloader_fake)},
            {"data_loader": src2_train_dataloader_fake, "iterator": iter(src2_train_dataloader_fake)},
            {"data_loader": src3_train_dataloader_fake, "iterator": iter(src3_train_dataloader_fake)}
        ]

        return real_loaders, fake_loaders, tgt_valid_dataloader

    def train(self, pe_method: str = 'base', hfn_method: str = 'base'):
        """
        학습 시작
        :param pe_method: 'base', 'maml', 'reptile'
        :param hfn_method: 'base', 'maml', 'reptile'
        """
        logging.info(f"Training pattern extractor with {pe_method}")
        if pe_method == 'base':
            pattern_extractor = self.meta_train()  
        elif pe_method in ['maml', 'reptile']:
            if self.config.TRAIN.PRETRAIN_HFN:
                hfn = self.pretrain_hfn(0)
                # pass
            if hfn is None:
                hfn = MetaHierachicalFusionNetwork(
                    mean_std_normalize=self.config.MODEL.MEAN_STD_NORMAL,
                    dropout_rate=self.config.TRAIN.DROPOUT
                ).cuda()

            logging.info(pe_method)

            pattern_extractor = self._train_pe_with_custom_method('pe', pe_method, hfn)
        else:
            raise ValueError(f"Unknown pattern extractor method: {pe_method}")
        
        logging.info(f"Training HFN with {hfn_method}")
        if hfn_method == 'base':
            hfn = self.train_hfn_from_scratch()  
        elif hfn_method in ['maml', 'reptile']:
            hfn = self._train_hfn_with_custom_method('hfn', hfn_method, pattern_extractor)
        else:
            raise ValueError(f"Unknown HFN method: {hfn_method}")
        
    def pretrain_hfn(self, data_num):
        """
        HFN 사전 학습 함수
        :param data_num: 랜덤하게 선택한 데이터셋 번호
        """
        logging.info(f"Pretraining HFN on dataset {data_num}")
        pattern_extractor = MetaPatternExtractor().cuda()
        hfn = MetaHierachicalFusionNetwork(
            mean_std_normalize=self.config.MODEL.MEAN_STD_NORMAL,
            dropout_rate=self.config.TRAIN.DROPOUT
        ).cuda()
        hfn_optimizer = optim.Adam(hfn.parameters(), lr=self.config.TRAIN.INIT_LR)

        # 손실 함수 초기화
        criterion_depth = nn.MSELoss()
        criterion_cls = nn.CrossEntropyLoss()
        for iter_num in range(250):
            # 랜덤한 데이터셋으로 task_data 로드
            task_data = get_data_from_loader_list(
                [self.real_loaders[data_num]],
                [self.fake_loaders[data_num]]
            )
            images = task_data[0].cuda()
            labels = task_data[1].cuda()
            depths = [d.cuda() for d in task_data[2]]

            # Forward pass
            with torch.no_grad():
                img_colored, _ = pattern_extractor(images)  # Pattern Extractor의 출력을 고정

            depth_pred, cls_preds = hfn(images, img_colored)  # HFN Forward

            # depth_pred와 cls_preds가 리스트일 경우 첫 번째 요소를 가져옴
            if isinstance(depth_pred, list):
                depth_pred = depth_pred[0]
            if isinstance(cls_preds, list):
                cls_preds = cls_preds[0]

            # 손실 계산
            depth_loss = criterion_depth(depth_pred.squeeze(), depths[0])
            cls_loss = criterion_cls(cls_preds, labels)
            total_loss = depth_loss + cls_loss

            # 정확도 계산
            _, predicted = torch.max(cls_preds, 1)  # 가장 높은 확률을 가진 클래스 선택
            correct = (predicted == labels).sum().item()
            acc = correct / labels.size(0)


            # Backpropagation
            hfn_optimizer.zero_grad()
            total_loss.backward()
            hfn_optimizer.step()

            # Logging
            if iter_num % 25 == 0:
                logging.info(f"Iteration {iter_num}: Pretrain Loss = {total_loss.item():.4f}, Accuracy = {acc:.4f}")


        checkpoint_data = {
                    'hfn_state_dict': hfn.state_dict(),
                    'hfn_optimizer_state_dict': hfn_optimizer.state_dict(),
                }
        checkpoint_path = f"{self.config.OUTPUT_DIR}/hfn_pretrained.pth"
        torch.save(checkpoint_data, checkpoint_path)
        logging.info(f"Pretrained HFN checkpoint saved to {checkpoint_path}")
        return hfn

    def _train_pe_with_custom_method(self, module_type: str, method: str, hfn):
        # if f"{module_type}_{method}" == 'pe_base':
        #     pattern_extractor = self.meta_train()
        if f"{method}" == 'maml':
            pattern_extractor = PatternExtractorMAMLTrainer(self.config, hfn).train()
        elif f"{method}" == 'reptile':
            pattern_extractor = PatternExtractorReptileTrainer(self.config, hfn).train()
        else:
            raise ValueError(f"Unknown method: {module_type}_{method}")
        logging.info(f"Training {module_type.upper()} with {method.upper()} method")

        return pattern_extractor
    

    def _train_hfn_with_custom_method(self, module_type: str,method: str, pattern_extractor):
        # if f"{method}" == 'hfn_base':
        #     hfn = self.train_hfn_from_scratch(pattern_extractor)
        if f"{method}" == 'maml':
            hfn = HFNMAMLTrainer(self.config, pattern_extractor).train()
        elif f"{method}" == 'reptile':
            hfn = HFNReptileTrainer(self.config, pattern_extractor).train()
        else:
            raise ValueError(f"Unknown method: {module_type}_{method}")
        return hfn
    # def _train_with_custom_method(self, module_type: str, method: str):
    #     # self.trainers = {
    #     #     'pe_maml': PatternExtractorMAMLTrainer(self.config, self.hfn),
    #     #     'pe_reptile': PatternExtractorReptileTrainer(self.config, self.hfn),
    #     #     'hfn_maml': HFNMAMLTrainer(self.config, self.pattern_extractor),
    #     #     'hfn_reptile': HFNReptileTrainer(self.config, self.pattern_extractor),
    #     #     # 'pe_base': self.meta_train(),
    #     #     # 'hfn_base': self.train_hfn_from_scratch(self.pattern_extractor)
    #     # }
    #     if f"{module_type}_{method}" == 'pe_base':
    #         pattern_extractor = self.meta_train()
    #     if f"{module_type}_{method}" == 'pe_maml':
    #         pattern_extractor = PatternExtractorMAMLTrainer(self.config, self.hfn).train()
    #     if f"{module_type}_{method}" == 'pe_reptile':
    #         pattern_extractor = PatternExtractorReptileTrainer(self.config, self.hfn).train()
    #     if f"{module_type}_{method}" == 'hfn_base':
    #         self.hfn = self.train_hfn_from_scratch(pattern_extractor)
    #     if f"{module_type}_{method}" == 'hfn_maml':
    #         HFNMAMLTrainer(self.config, pattern_extractor).train()
    #     if f"{module_type}_{method}" == 'hfn_reptile':
    #         HFNReptileTrainer(self.config, pattern_extractor).train()
    #     # trainer = self.trainers[f"{module_type}_{method}"]
    #     logging.info(f"Training {module_type.upper()} with {method.upper()} method")
        # trainer.train()
    # def _train_with_custom_method(self, module_type: str, method: str):
    #     if f"{module_type}_{method}" in ['pe_maml', 'pe_reptile']:
    #         self.pattern_extractor = (
    #             PatternExtractorMAMLTrainer(self.config, self.hfn).train()
    #             if method == 'maml'
    #             else PatternExtractorReptileTrainer(self.config, self.hfn).train()
    #         )
    #     if f"{module_type}_{method}" in ['hfn_maml', 'hfn_reptile']:
    #         if not self.pattern_extractor:
    #             raise ValueError("Pattern extractor must be initialized before HFN training.")
    #         trainer = HFNMAMLTrainer if method == 'maml' else HFNReptileTrainer
    #         trainer(self.config, self.pattern_extractor).train()
    #     else:
    #         raise ValueError(f"Unknown method: {module_type}_{method}")
    #     logging.info(f"Training {module_type.upper()} with {method.upper()} method")

    def _validate_and_save(self, iter_num: int):
        with torch.no_grad():
            val_loss = self.validate(iter_num, self.val_loader)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint()

        logging.info(f"Validation Loss: {val_loss:.4f}")

    def save_checkpoint(self, state = None):
        """
        Best Checkpoint 저장
        """
        if state is None:

            state = {
                'pattern_extractor': self.pattern_extractor.state_dict(),
                'hfn': self.hfn.state_dict(),
                'val_loss': self.best_val_loss
            }
        torch.save(state, f"{self.config.OUTPUT_DIR}/best_model.pth")