import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import ConcatDataset, random_split
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from torch.utils.data import Dataset

def get_data(args):
    torch.manual_seed(args.seed)
    if args.dataset == 'cifar10':
        args.mean = (0.4914, 0.4822, 0.4465)
        args.std = (0.2023, 0.1994, 0.2010)
        if args.sota:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(args.mean, args.std),
            ])
            transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(args.mean, args.std),
                ])
        else:
            if 'mlp' not in args.model:
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(args.mean, args.std),
                ])
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(args.mean, args.std),
                ])
            else:
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                ])
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                ])
        
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    elif args.dataset == 'imagenet':
        args.mean = (0.485, 0.456, 0.406)
        args.std = (0.229, 0.224, 0.225)
        
        if args.sota:
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(args.mean, args.std),
            ])
            transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(args.mean, args.std),
                ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(args.mean, args.std),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(args.mean, args.std),
            ])
        
        trainset = datasets.ImageFolder('data/tiny-imagenet-200/train/', transform_train)
        testset = datasets.ImageFolder('data/tiny-imagenet-200/val/images/', transform_test)

    elif args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True)
        x = np.concatenate([np.asarray(trainset[i][0]) for i in range(len(trainset))])
        args.mean = (np.mean(x, axis=(0, 1))/255).tolist()
        args.std = (np.std(x, axis=(0, 1))/255).tolist()
        
        if args.sota:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(args.mean, args.std),
            ])
            transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(args.mean, args.std),
                ])
        else:
            if 'mlp' not in args.model:
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(args.mean, args.std),
                ])
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(args.mean, args.std),
                ])
            else:
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                ])
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                ])
        
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    elif args.dataset == 'svhn':
        trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True)
        x = np.concatenate([np.asarray(trainset[i][0]) for i in range(len(trainset))])
        args.mean = (np.mean(x, axis=(0, 1))/255).tolist()
        args.std = (np.std(x, axis=(0, 1))/255).tolist()
        
        if 'mlp' not in args.model:
            trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(args.mean, args.std)
            ])
        else:
            trans = transforms.Compose([
                transforms.ToTensor(),
            ])

        trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=trans)
        testset = torchvision.datasets.SVHN(root='./data',  split='test', download=True, transform=trans)        
        trainset, _ = torch.utils.data.random_split(trainset, [50000, len(trainset) - 50000])

    elif args.dataset == 'fashion_mnist':
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True)
        x = np.concatenate([np.asarray(trainset[i][0]) for i in range(len(trainset))])
        args.mean = (np.mean(x, axis=(0, 1))/255).tolist()
        args.std = (np.std(x, axis=(0, 1))/255).tolist()

        if 'mlp' not in args.model:
            trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(args.mean, args.std),
            ])
        else:
            trans = transforms.Compose([
                transforms.ToTensor(),
            ])
        
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=trans)
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=trans)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)
    
    return trainset, train_loader, testset, test_loader

class Data_transform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)
