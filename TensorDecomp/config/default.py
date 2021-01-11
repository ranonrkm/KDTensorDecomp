import os
from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# CUDNN
# -----------------------------------------------------------------------------
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = 'resnet34'
_C.MODEL.CKPT = '/scratch/16ee35016/checkpoints/cifar100/Undecomposedresnet34.pth'
_C.MODEL.CKPT_ROOT = '/scratch/16ee35016/checkpoints'
_C.MODEL.DECOMP = 'tucker'
_C.MODEL.DECOMPOSE_CKPT = ''

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.NAME = 'ImageNet'
_C.DATASET.NUM_CLASSES = 1000


# -----------------------------------------------------------------------------
# Attack
# -----------------------------------------------------------------------------
_C.ATTACK = CN()
_C.ATTACK.NAME = 'deepfool'
_C.ATTACK.EPS = 0.01
_C.ATTACK.MAX_ITER = 200
_C.ATTACK.LR = 0.008


_C.SOLVER = CN()
_C.SOLVER.LOSS = 'L2'
_C.SOLVER.TEACHER_LOGITS = '/scratch/16ee35016/BTP/KDTensorDecomp/TensorDecomp/checkpoints/ImageNet/vgg16_bn/teacher_logits.pth.tar'
_C.SOLVER.KD_temp = 8
_C.SOLVER.LAMBA = 0.8
