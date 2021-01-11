from .resnet import resnet_book
from .vgg import vgg_book
from TensorDecomp.config import config

def is_resnet(name):
    name = name.lower()
    return name.startswith('resnet')

def is_vgg(name):
    name = name.lower()
    return name.startswith('vgg')

# TODO: mobilenets to be added
def is_mobile(name):
    name = name.lower()
    return name.startswith('mobilenet')


def get_network():
    
    name = config.MODEL.NAME
    if is_resnet(name):
        resnet_size = name[6:]
        model = resnet_book.get(resnet_size)(num_classes=config.DATASET.NUM_CLASSES)
    elif is_vgg(name):
        vgg_size = name[3:]
        model = vgg_book.get(vgg_size)(num_classes=config.DATASET.NUM_CLASSES)
    else:
        raise NotImplementedError('{} model not implemented.'.format(name))
    return model


        
