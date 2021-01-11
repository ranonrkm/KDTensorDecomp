import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from TuckerDecomp.utils import save_checkpoint
from TuckerDecomp.config import config

def saveLogits(data_loader, model, args):

    #save logits in dictionary
    logit_dict = torch.zeros([len(data_loader), config.DATASET.NUM_CLASSES])

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, targets, index) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)

            output = model(images)
            for k in range(len(index)):
                idx = int(index[k].item())
                logit_dict[idx] = output[k]
            
    save_checkpoint({'logit_dict': logit_dict}, 
                    filename=os.path.join('./checkpoint/'config.DATASET.NAME,config.MODEL.NAME,'teacher_logits.pth.tar'))
