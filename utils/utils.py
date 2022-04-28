import torch
import math
import time
import os


def mkdirs(dir_):
    if os.path.exists(dir_):
        key = input("{} existed. Rewrite (y/N)?".format(dir_))
        if key.lower() != 'y':
            dir_ = dir_ + time.strftime("-%m%d%H%M")
    os.makedirs(dir_, exist_ok=True)
    return dir_


def log_density(x, mu, std, logstd):
    var = std.pow(2)
    log_density = -(x - mu).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - logstd
    return log_density.sum(1, keepdim=True)



def flat_grad(grads):
    grad_flatten = []
    for grad in grads:
        grad_flatten.append(grad.view(-1))
    grad_flatten = torch.cat(grad_flatten)
    return grad_flatten


def flat_hessian(hessians):
    hessians_flatten = []
    for hessian in hessians:
        hessians_flatten.append(hessian.contiguous().view(-1))
    hessians_flatten = torch.cat(hessians_flatten).data
    return hessians_flatten


def flat_params(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    params_flatten = torch.cat(params)
    return params_flatten


def update_model(model, new_params):
    index = 0
    for params in model.parameters():
        params_length = len(params.view(-1))
        new_param = new_params[index: index + params_length]
        new_param = new_param.view(params.size())
        params.data.copy_(new_param)
        index += params_length

class AverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """
    def __init__(self):
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


def parse_label_from_video_name(video_name):
    """

    :param video_name: CASIA/OULU/REPLAY/MSU /*.avi
    :return: label: 0-real face, 1-print face, 2-display, 3-mask
    """

