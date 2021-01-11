import os
import torch
import torchvision
import torchvision.transforms as transforms
from TensorDecomp.config import config

data_path = os.path.join(os.environ['DATA_PATH'], 'cifar')

class indexedTrainset(torch.utils.data.Dataset):
    def __init__(self):
        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        
        if config.DATASET.NAME.lower() == 'cifar100':
            self.trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_train)
        else:
            self.trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
        
    def __getitem__(self, index):
        data, target = self.trainset[index]
        return data, target, index

    def __len__(self):
        return len(self.trainset)

def DataLoader(args):
    transform_test = transforms.Compose([
        transforms.Resize(32),transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    trainset = indexedTrainset()
    if config.DATASET.NAME.lower() == 'cifar100':
        testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform_test)
    else:
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
        
    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=args.batch_size, 
                                              shuffle=True, 
                                              num_workers=args.workers)
    testloader = torch.utils.data.DataLoader(testset, 
                                             batch_size=args.batch_size,
                                             shuffle=False, 
                                             num_workers=args.workers) 
    return trainloader, testloader
