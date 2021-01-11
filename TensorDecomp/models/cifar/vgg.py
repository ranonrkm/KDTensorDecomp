import torch
import torch.nn as nn
from TensorDecomp.tools import decompositions
from TensorDecomp.config import config

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x1 = self.features(x)
        out = self.classifier(x1.view(x1.size(0), -1))
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def decompose(self):
        
        N = len(self.features._modules.keys())
        k = -1
        for i, key in enumerate(self.features._modules.keys()):
            if i >= N-2:
                break
            if isinstance(self.features._modules[key], nn.modules.conv.Conv2d):
                k += 1
                if k == 0:
                    continue
                conv_layer = self.features._modules[key]
                if config.MODEL.DECOMP == 'cp':
                    decomposed = decompositions.cp_decomposition_conv_layer(conv_layer)
                else:
                    decomposed = decompositions.tucker_decomposition_conv_layer(conv_layer)
                self.features._modules[key] = decomposed



def VGG11(num_classes=10):
    return VGG('VGG11', num_classes)

def VGG13(num_classes=10):
    return VGG('VGG13', num_classes)

def VGG16(num_classes=10):
    return VGG('VGG16', num_classes)

def VGG19(num_classes=10):
    return VGG('VGG19', num_classes)

vgg_book = {
    '11': VGG11,
    '13': VGG13,
    '16': VGG16,
    '19': VGG19,
}
