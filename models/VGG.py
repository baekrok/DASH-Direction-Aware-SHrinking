import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from einops import rearrange, repeat

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

class VGG(nn.Module):
    def __init__(self, batch_norm, args, cfg, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        
        if args.dataset == 'fashion_mnist':
            in_channels = 1
            padding = 1
        else:
            in_channels = 3
            padding = 0
        i, in_channels, self.layer1 = make_layers(cfgs[cfg], batch_norm, 0, in_channels)
        i, in_channels, self.layer2 = make_layers(cfgs[cfg], batch_norm, i+1, in_channels)
        i, in_channels, self.layer3 = make_layers(cfgs[cfg], batch_norm, i+1, in_channels, padding)
        i, in_channels, self.layer4 = make_layers(cfgs[cfg], batch_norm, i+1, in_channels)
        _, _, self.layer5 = make_layers(cfgs[cfg], batch_norm, i+1, in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self.random_init(args)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x = self.avgpool(x5)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def random_init(self, args):
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def fc_reset(self, args, exp):
        torch.manual_seed(args.seed * exp)
        torch.cuda.manual_seed_all(args.seed * exp)
        
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
                m.reset_parameters()
        
        self.classifier.apply(weight_reset)
        
        
    def weight_shrink(self, args):
        def shrink(m):
            temp_m = copy.deepcopy(m)
            with torch.no_grad():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    m.weight.mul_(args.sp_lambda)
                    if m.bias != None:
                        m.bias.mul_(args.sp_lambda)
        model = self.apply(shrink)
                
        
    def weight_sp(self, args):
        def shrink_perturb(m):
            temp_m = copy.deepcopy(m)
            with torch.no_grad():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    weight = torch.normal(mean = torch.zeros(temp_m.weight.size()), std = torch.tensor(0.01).repeat(temp_m.weight.size())).to(args.device)
                    if temp_m.bias != None:
                        bias = torch.normal(mean = torch.zeros(temp_m.bias.size()), std = torch.tensor(0.01).repeat(temp_m.bias.size())).to(args.device)
                        
                    m.weight.mul_(args.sp_lambda).add_(weight)
                    if m.bias != None:
                        m.bias.mul_(args.sp_lambda).add_(bias)
        model = self.apply(shrink_perturb)
        

def make_layers(cfg, batch_norm=False, index=0, in_channels=3, padding=0):
    layers = []
    
    for i in range(index, len(cfg)):
        if cfg[i] == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, padding=padding)]
            return i, in_channels, nn.Sequential(*layers)
        else:
            conv2d = nn.Conv2d(in_channels, cfg[i], kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(cfg[i]), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = cfg[i]
    



cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(args, arch, cfg, batch_norm, num_classes=10):
    model = VGG(batch_norm, args, cfg, num_classes)
    return model


def vgg11(args, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg(args, 'vgg11', 'A', False, **kwargs)



def vgg11_bn(args, **kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg(args, 'vgg11_bn', 'A', True, **kwargs)



def vgg13(args, **kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg(args, 'vgg13', 'B', False, **kwargs)



def vgg13_bn(args, **kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg(args, 'vgg13_bn', 'B', True, **kwargs)



def vgg16(args):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if args.dataset == 'cifar10':
        return _vgg(args, 'vgg16', 'D', False)
    elif args.dataset == 'imagenet':
        return _vgg(args, 'vgg16', 'D', False, 200)


def vgg16_bn(args):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if args.dataset == 'cifar10':
        return _vgg(args, 'vgg16_bn', 'D', True)
    elif args.dataset == 'imagenet':
        return _vgg(args, 'vgg16_bn', 'D', True, 200)
    elif args.dataset == 'cifar100':
        return _vgg(args, 'vgg16_bn', 'D', True, 100)
    elif args.dataset == 'fashion_mnist':
        return _vgg(args, 'vgg16_bn', 'D', True, 10)
    elif args.dataset == 'svhn':
        return _vgg(args, 'vgg16_bn', 'D', True, 10)
    
def vgg19(args, **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg(args, 'vgg19', 'E', False, **kwargs)



def vgg19_bn(args, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg(args, 'vgg19_bn', 'E', True, **kwargs)
