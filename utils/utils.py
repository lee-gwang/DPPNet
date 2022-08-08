from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import logging
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from utils.rcp_utils import DPPConv2d


class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)
def print_keep_ratio(model, logger):
    total = 0.
    keep = 0.
    for layer in model.modules():
        if isinstance(layer, DWPConv2d):
            w_shape = layer.weight.shape
            abs_weight = torch.abs(layer.weight).view(w_shape[0], -1).mean(1)
            threshold = layer.threshold
            mask = layer.step(abs_weight-threshold)
            k = torch.sum(mask) ; t = mask.numel() # keep, total
            ratio = k / t
            total += t
            keep += k
            #logger.info("Layer threshold {:.4f}".format(layer.threshold[0]))
            logger.info(f"{layer}, keep ratio {ratio :.4f}, {int(k)}/{int(t)}")

        if isinstance(layer, nn.Conv2d):
            total += layer.weight.shape[0] # out_c, in_c, k, k
            keep += layer.weight.shape[0]

    #logger.info("Model keep ratio {:.4f}".format(keep/total))
    return #keep / total

def create_logger(save_path='', file_type='', level='debug'):

    if level == 'debug':
        _level = logging.DEBUG
    elif level == 'info':
        _level = logging.INFO

    logger = logging.getLogger()
    logger.setLevel(_level)

    cs = logging.StreamHandler()
    cs.setLevel(_level)
    logger.addHandler(cs)

    if save_path != '':
        file_name = os.path.join(save_path, file_type + '_log.txt')
        fh = logging.FileHandler(file_name, mode='w')
        fh.setLevel(_level)

        logger.addHandler(fh)

    return logger
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def save_models(model, path):
    """Save model to given path
    Args:
        model: model to be saved
        path: path that the model would be saved
        epoch: the epoch the model finished training
    """
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = os.path.join(path, "model_{}.pt".format(suffix))
    torch.save(model, file_path) #pwf file

def poly_decay_lr(optimizer, global_steps, total_steps, base_lr, end_lr, power):
    """Sets the learning rate to be polynomially decaying"""
    lr = (base_lr - end_lr) * (1 - global_steps/total_steps) ** power + end_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
