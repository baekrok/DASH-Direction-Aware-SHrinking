from torch import nn
import copy
import torch

class mlp(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        num_classes = 10
        if ('cifar10' in args.dataset) | (args.dataset == 'svhn'):
            self.fc1 = nn.Linear(3*32*32, args.width)
            if args.dataset == 'cifar100':
                num_classes = 100
        elif 'fashion_mnist' in args.dataset:
            self.fc1 = nn.Linear(1*28*28, args.width)
        self.activation1 = nn.ReLU()
        if args.depth == 2:
            self.fc2 = nn.Linear(args.width, num_classes)
        elif args.depth == 3:
            self.fc2 = nn.Linear(args.width, args.width)
            self.activation2 = nn.ReLU()
            self.fc3 = nn.Linear(args.width, num_classes)
            
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c*h*w)
        if self.args.depth == 2:
            return self.fc2(self.activation1(self.fc1(x)))
        elif self.args.depth == 3:
            return self.fc3(self.activation2(self.fc2(self.activation1(self.fc1(x)))))
        
    def weight_sp(self, args):
        def shrink_perturb(m):
            temp_m = copy.deepcopy(m)
            with torch.no_grad():
                if isinstance(m, nn.Linear):
                    weight = torch.normal(mean = torch.zeros(temp_m.weight.size()), std = torch.tensor(0.01).repeat(temp_m.weight.size())).to(args.device)
                    if temp_m.bias != None:
                        bias = torch.normal(mean = torch.zeros(temp_m.bias.size()), std = torch.tensor(0.01).repeat(temp_m.bias.size())).to(args.device)
                        
                    m.weight.mul_(args.sp_lambda).add_(weight)
                    if m.bias != None:
                        m.bias.mul_(args.sp_lambda).add_(bias)
        model = self.apply(shrink_perturb)
        
class mlp_bn(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        num_classes = 10
        if ('cifar10' in args.dataset) | (args.dataset == 'svhn'):
            self.fc1 = nn.Linear(3*32*32, args.width)
            if args.dataset == 'cifar100':
                num_classes = 100
        elif 'fashion_mnist' in args.dataset:
            self.fc1 = nn.Linear(1*28*28, args.width)
        elif 'imagenet' in args.dataset:
            self.fc1 = nn.Linear(3*64*64, args.width)
            num_classes = 200
        self.bn1 = nn.BatchNorm1d(args.width)
        self.activation1 = nn.ReLU()
        if args.depth == 2:
            self.fc2 = nn.Linear(args.width, num_classes)
        elif args.depth == 3:
            self.fc2 = nn.Linear(args.width, args.width)
            self.bn2 = nn.BatchNorm1d(args.width)
            self.activation2 = nn.ReLU()
            self.fc3 = nn.Linear(args.width, num_classes)
            
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c*h*w)
        if self.args.depth == 2:
            return self.fc2(self.activation1(self.bn1(self.fc1(x))))
        elif self.args.depth == 3:
            return self.fc3(self.activation2(self.bn2(self.fc2(self.activation1(self.bn1(self.fc1(x)))))))
        
    def weight_sp(self, args):
        def shrink_perturb(m):
            temp_m = copy.deepcopy(m)
            with torch.no_grad():
                if isinstance(m, nn.Linear):
                    weight = torch.normal(mean = torch.zeros(temp_m.weight.size()), std = torch.tensor(0.01).repeat(temp_m.weight.size())).to(args.device)
                    if temp_m.bias != None:
                        bias = torch.normal(mean = torch.zeros(temp_m.bias.size()), std = torch.tensor(0.01).repeat(temp_m.bias.size())).to(args.device)
                        
                    m.weight.mul_(args.sp_lambda).add_(weight)
                    if m.bias != None:
                        m.bias.mul_(args.sp_lambda).add_(bias)
        model = self.apply(shrink_perturb)
        
    def fc_reset(self, args, exp):
        torch.manual_seed(args.seed * exp)
        torch.cuda.manual_seed_all(args.seed * exp)
        
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
                m.reset_parameters()
        
        self.fc3.apply(weight_reset)