from models.ResNet import resnet18
from models.ResNet_SoTA import resnet18_sota
from models.MLP import *
from models.VGG import *

model_factories = {
    'resnet18': resnet18,
    'resnet18_sota': resnet18_sota,
    'mlp': mlp_bn,
    'vgg16': vgg16_bn, 
}

def get_available_models():
    return model_factories.keys()

def get_model(args):
    if args.sota == True:
        args.model = 'resnet18_sota'
    return model_factories[args.model](args)
