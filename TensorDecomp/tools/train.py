import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import sys
from TensorDecomp.config import config
from TensorDecomp.utils import *
CUDA_LAUNCH_BLOCKING=1


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    if args.teacher:
        try:
            checkpoint = torch.load(config.SOLVER.TEACHER_LOGITS)['logit_dict']
        except FileNotFoundError:
            sys.exit("Teacher logits not found")
    
    for i, (images, targets, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        targets = targets.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)

        if args.teacher:
            teacher_logits = torch.zeros_like(output)
            for k in range(len(index)):
                idx = int(index[k].item())
                teacher_logits[k] = checkpoint[idx]
    
            if config.SOLVER.LOSS == 'KD':
                loss_SL = CE_criterion(output, targets)
                T = config.SOLVER.KD_temp
                loss_KD = criterion(F.log_softmax(output/T, dim=1),
                                    F.softmax(teacher_logits/T, dim=1))
                loss = (1 - config.SOLVER.LAMBDA) * loss_SL + config.SOLVER.LAMBDA * T * T * loss_KD
            else:
                loss = criterion(teacher_logits, output)
        else:
            loss = criterion(output, targets)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, targets) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, targets)
            acc1, acc5 = accuracy(output, targets, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    return top1.avg, top5.avg

