"""
    Ref: https://github.com/jacobgil/pytorch-tensor-decompositions
"""


import tensorly as tl
from tensorly.decomposition import parafac, partial_tucker
import numpy as np
import torch
import torch.nn as nn
from .VBMF import VBMF
from TensorDecomp.config import config

tl.set_backend('pytorch')

def cp_decomposition_conv_layer(layer, rank=None):
    """ Gets a conv layer and a target rank, 
        returns a nn.Sequential object with the decomposition """
    device = layer.weight.device
    # Perform CP decomposition on the layer weight tensorly. 
    if rank == None:
        ranks = estimate_ranks(layer)
        rank = max(ranks)
        print(layer,"estimated ranks are",ranks,"ranks accepted",rank)

    last, first, vertical, horizontal = \
        parafac(layer.weight.data.cpu(), rank=rank, init='random') #changing "svd" -> "random"

    pointwise_s_to_r_layer = torch.nn.Conv2d(in_channels=first.shape[0], \
            out_channels=first.shape[1], kernel_size=1, stride=1, padding=0,
            dilation=layer.dilation, bias=False)

    depthwise_vertical_layer = torch.nn.Conv2d(in_channels=vertical.shape[1],
            out_channels=vertical.shape[1], kernel_size=(vertical.shape[0], 1),
            stride=1, padding=(layer.padding[0], 0), dilation=layer.dilation,
            groups=vertical.shape[1], bias=False)

    depthwise_horizontal_layer = \
        torch.nn.Conv2d(in_channels=horizontal.shape[1], \
            out_channels=horizontal.shape[1],
            kernel_size=(1, horizontal.shape[0]), stride=layer.stride,
            padding=(0, layer.padding[0]),
            dilation=layer.dilation, groups=horizontal.shape[1], bias=False)

    pointwise_r_to_t_layer = torch.nn.Conv2d(in_channels=last.shape[1], \
            out_channels=last.shape[0], kernel_size=1, stride=1,
            padding=0, dilation=layer.dilation, bias=True)

    if layer.bias is not None:
        pointwise_r_to_t_layer.bias.data = layer.bias.data.to(device)

    depthwise_horizontal_layer.weight.data = \
        torch.transpose(horizontal, 1, 0).unsqueeze(1).unsqueeze(1).to(device)
    depthwise_vertical_layer.weight.data = \
        torch.transpose(vertical, 1, 0).unsqueeze(1).unsqueeze(-1).to(device)
    pointwise_s_to_r_layer.weight.data = \
        torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1).to(device)
    pointwise_r_to_t_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1).to(device)

    new_layers = [pointwise_s_to_r_layer, depthwise_vertical_layer, \
                    depthwise_horizontal_layer, pointwise_r_to_t_layer]

    return nn.Sequential(*new_layers)


def estimate_ranks(layer):
    """ Unfold the 2 modes of the Tensor the decomposition will 
    be performed on, and estimates the ranks of the matrices using VBMF 
    """
    weights = layer.weight.data
    unfold_0 = tl.base.unfold(weights, 0)
    unfold_1 = tl.base.unfold(weights, 1)
    _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
    _, diag_1, _, _ = VBMF.EVBMF(unfold_1)
    ranks = [diag_0.shape[0], diag_1.shape[1]]
    return ranks

def tucker_decomposition_conv_layer(layer):
    """ Gets a conv layer, 
        returns a nn.Sequential object with the Tucker decomposition.
        The ranks are estimated with a Python implementation of VBMF
        https://github.com/CasvandenBogaard/VBMF
    """
    ranks = estimate_ranks(layer)
    print(layer, "VBMF Estimated ranks", ranks)
    core, [last, first] = \
        partial_tucker(layer.weight.data, \
            modes=[0, 1], ranks=ranks, init='svd')   #changing random from "svd"

    # A pointwise convolution that reduces the channels from S to R3
    first_layer = torch.nn.Conv2d(in_channels=first.shape[0], \
            out_channels=first.shape[1], kernel_size=1,
            stride=1, padding=0, dilation=layer.dilation, bias=False)

    # A regular 2D convolution layer with R3 input channels 
    # and R3 output channels
    core_layer = torch.nn.Conv2d(in_channels=core.shape[1], \
            out_channels=core.shape[0], kernel_size=layer.kernel_size,
            stride=layer.stride, padding=layer.padding, dilation=layer.dilation,
            bias=False)

    bias = False
    if layer.bias is not None:
        bias = True
    # A pointwise convolution that increases the channels from R4 to T
    last_layer = torch.nn.Conv2d(in_channels=last.shape[1], \
        out_channels=last.shape[0], kernel_size=1, stride=1,
        padding=0, dilation=layer.dilation, bias=bias)

    if bias:
        last_layer.bias.data = layer.bias.data

    first_layer.weight.data = \
        torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = core

    new_layers = [first_layer, core_layer, last_layer]
    return nn.Sequential(*new_layers)

def decompose_vgg(net):
    N = len(net.features._modules.keys())
    k = -1
    for i, key in enumerate(net.features._modules.keys()):
        if i >= N-2:
            break
        if isinstance(net.features._modules[key], nn.modules.conv.Conv2d):
            k += 1
            if k == 0:
                continue
            conv_layer = net.features._modules[key]
            if config.MODEL.DECOMP == 'cp':
                decomposed = cp_decomposition_conv_layer(conv_layer)
            else:
                decomposed = tucker_decomposition_conv_layer(conv_layer)
            net.features._modules[key] = decomposed
    
    return net

def decompose_resnet(net):
    for name, m in net.named_children():
        num_children = sum(1 for i in m.children())
        if num_children != 0:
            layer = getattr(net, name)
            # decomp every block (basicblock or bottleneck)
            for i in range(num_children):
                block = layer[i]
                conv1 = getattr(block, 'conv1')
                conv2 = getattr(block, 'conv2')
                new_conv1 = tucker_decomposition_conv_layer(conv1)
                new_conv2 = tucker_decomposition_conv_layer(conv2)
                setattr(block, 'conv1', nn.Sequential(*new_conv1))
                setattr(block, 'conv2', nn.Sequential(*new_conv2))
                del conv1
                del conv2
                del block
            del layer

    return net

def decompose(net):

    name = config.MODEL.NAME
    if name.lower().startswith('resnet'):
        return decompose_resnet(net)
    elif name.lower().startswith('vgg'):
        return decompose_vgg(net)
    else:
        raise NotImplementedError('{} architecture not implemented yet'.format(name))
     
