import argparse
import os
import sys
import random
import time
import warnings

import torch
import torch.nn as nn
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim import lr_scheduler

from datasets import cifar, ImageNet
import models.cifar as cifarmodels
from tools.train import train, validate
from tools.decompositions import decompose
from utils import *
from TensorDecomp.config import config
#from RAdam.cifar_imagenet.utils.radam import RAdam, RAdam_4step, AdamW

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',default='../../avinab/ImagenetData',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--decompose', dest='decompose', action='store_true',
                    help='network is to be decomposed')
parser.add_argument('--teacher', dest='teacher', action='store_true',
                    help='there is a teacher model for reference')


best_acc1 = 0
best_acc5 = 0

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    main_worker(args.gpu, args)


def main_worker(gpu, args):
    global best_acc1
    global best_acc5
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if config.DATASET.NAME.lower() == 'imagenet':
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(config.MODEL.NAME))
            model = models.__dict__[config.MODEL.NAME](pretrained=True)
        else:
            print("=> creating model '{}'".format(config.MODEL.NAME))
            model = models.__dict__[config.MODEL.NAME]()
    elif config.DATASET.NAME.lower().startswith('cifar'):
        model = cifarmodels.get_network()
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(config.MODEL.NAME))
            model.load_state_dict(torch.load(config.MODEL.CKPT))

    if args.decompose:
        model = decompose(model)
        if os.path.isfile(config.MODEL.DECOMPOSE_CKPT):
            model.load_state_dict(torch.load(config.MODEL.DECOMPOSE_CKPT))

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)


    # define loss function (criterion) and optimizer
    if args.teacher:
        if config.SOLVER.LOSS == 'L2':
            criterion = nn.MSELoss(reduction='sum').cuda(args.gpu)
        elif config.SOLVER.LOSS == 'L1':
            criterion = nn.L1Loss(reduction='sum').cuda(args.gpu)
        elif config.SOLVER.LOSS == 'KD':
            criterion = nn.KLDivLoss().cuda(args.gpu)
        else:
            criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    else:
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
#    optimizer = RAdam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    LR_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=6)
    #LR_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30,50,80], gamma=0.1)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    
    if config.DATASET.NAME.lower() == 'imagenet': 
        train_loader, val_loader = ImageNet.DataLoader(args)
    else:
        train_loader, val_loader = cifar.DataLoader(args)

    if args.evaluate:
        validate(val_loader, model, criterion, args)  
        sys.exit(0)	
        return

    for epoch in range(args.start_epoch, args.epochs):
        #adjust_learning_rate(optimizer, epoch, args)
        for param_group in optimizer.param_groups:
                print("learning rate:",param_group['lr'])

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)
        # evaluate on validation set
        acc1,acc5 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)
        LR_scheduler.step(best_acc1)
        print(best_acc1, best_acc5)
 
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': config.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
        for param_group in optimizer.param_groups:
                print("learning rate:",param_group['lr'])
                if param_group['lr'] < 1e-12:
                    print(epoch)
                    print("------------training ends-------------")
                    sys.exit(0)



if __name__ == '__main__':
    main()
