import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.optim.lr_scheduler import _LRScheduler
from torch.autograd import Variable
from collections import OrderedDict
from sklearn.neighbors import NearestNeighbors
import random
import shutil
import time
import warnings
import datetime
from TensorDecomp.config import config

x = datetime.datetime.now()

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_checkpoint(state, is_best=False, filename=None):
    if filename is None:
        filename = os.path.join('./checkpoints', config.DATASET.NAME, config.MODEL.NAME, 'checkpoint-'+x.strftime("%d-%m")+'.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join('./checkpoints', config.DATASET.NAME, config.MODEL.NAME,'model-best-'+x.strftime("%d-%m")+'pth.tar'))


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
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
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


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]



def density_entropy(X):
    K = 5
    N, C, D = X.shape
    x = X.transpose(1, 0, 2).reshape(C, N, -1)
    score = []
    for c in range(C):
        nbrs = NearestNeighbors(n_neighbors=K + 1).fit(x[c])
        dms = []
        for i in range(N):
            dm = 0
            dist, ind = nbrs.kneighbors(x[c, i].reshape(1, -1))
            for j, id in enumerate(ind[0][1:]):
                dm += dist[0][j + 1]

            dms.append(dm)

        dms_sum = sum(dms)
        en = 0
        for i in range(N):
            en += -dms[i]/dms_sum*math.log(dms[i]/dms_sum, 2)

        score.append(en)
    return np.array(score)


def info_richness(model, conv_list):
    model.eval()
    for i in conv_list:
        weight = model.features[i].weight.data.cpu()
        in_channels, out_channels,_,_ = weight.shape
        weight = weight.numpy().transpose(1, 0, 2, 3)
        entropy = density_entropy(weight.reshape(out_channels, in_channels, -1))
        print(model.features[i])
        print(max(entropy),min(entropy))

# Function to return Number of parameters & Multiplications
def summary(model, input_size):
    #inspired from https://github.com/sksq96/pytorch-summary/blob/master/torchsummary/torchsummary.py
        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split('.')[-1].split("'")[0]
                module_idx = len(summary)
                if module_idx == 261:
                    print(str(module.__class__))

                m_key = '%s-%i' % (class_name, module_idx+1)
                summary[m_key] = OrderedDict()
                summary[m_key]['input_shape'] = list(input[0].size())
                summary[m_key]['input_shape'][0] = -1
                if isinstance(output, (list,tuple)):
                    summary[m_key]['output_shape'] = [[-1] + list(o.size())[1:] for o in output]
                else:
                    summary[m_key]['output_shape'] = list(output.size())
                    summary[m_key]['output_shape'][0] = -1

                wt_params = 0
                if hasattr(module, 'weight'):
                    wt_params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]['trainable'] = module.weight.requires_grad
                bias_params = 0
                if hasattr(module, 'bias') and hasattr(module.bias, 'size'):
                    bias_params +=  torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]['nb_params'] = wt_params + bias_params
                
                if 'conv' in m_key.lower():
                    summary[m_key]['MulC'] = wt_params * summary[m_key]['output_shape'][-1] * summary[m_key]['output_shape'][-2]
                elif 'linear' in m_key.lower():
                    summary[m_key]['MulC'] = wt_params
                else:
                    summary[m_key]['MulC'] = 0

            if (not isinstance(module, nn.Sequential) and 
               not isinstance(module, nn.ModuleList) and 
               not (module == model)):
                hooks.append(module.register_forward_hook(hook))
                
        if torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor
        
        # check if there are multiple inputs to the network
        if isinstance(input_size[0], (list, tuple)):
            x = [Variable(torch.rand(1,*in_size)).type(dtype) for in_size in input_size]
        else:
            x = Variable(torch.rand(1,*input_size)).type(dtype)
            
            
        # print(type(x[0]))
        # create properties
        summary = OrderedDict()
        hooks = []
        # register hook
        model.apply(register_hook)
        # make a forward pass
        # print(x.shape)
        model.cuda()
        print(next(model.parameters()).is_cuda)
        model(x)
        # remove these hooks
        for h in hooks:
            h.remove()

        #print('----------------------------------------------------------------')
        line_new = '{:>20}  {:>25} {:>15} {:>20}'.format('Layer (type)', 'Output Shape', 'Param #', 'Muls #')
        #print(line_new)
        #print('======================================================================================')
        total_params = 0
        trainable_params = 0
        total_comp = 0
        for layer in summary:
            # input_shape, output_shape, trainable, nb_params
            line_new = '{:>20}  {:>25} {:>15} {:>20}'.format(layer, str(summary[layer]['output_shape']), summary[layer]['nb_params'], summary[layer]['MulC'])
            total_params += summary[layer]['nb_params']
            total_comp   += summary[layer]['MulC']
            if 'trainable' in summary[layer]:
                if summary[layer]['trainable'] == True:
                    trainable_params += summary[layer]['nb_params']
            #print(line_new)
        #print('======================================================================================')
        #print('Total params: ' + str(total_params))
        #print('Trainable params: ' + str(trainable_params))
        #print('Non-trainable params: ' + str(total_params - trainable_params))
        #print('Total Multiplications: ' + str(total_comp))
        #print('----------------------------------------------------------------')        
        #line_new = '{:>25}  {:>25}'.format(str(trainable_params.numpy()), str(total_comp.numpy()))
        #print(line_new)
        #return trainable_params, total_comp
        return trainable_params, total_comp
