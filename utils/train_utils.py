import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from einops import rearrange
from torch import optim
import torchvision.transforms as transforms
import numpy as np
import random
from torch.utils.data import ConcatDataset, random_split, DataLoader
from utils.SAM import SAM
from torch.nn.parameter import Parameter, UninitializedParameter
from torch import _weight_norm, norm_except_dim
from typing import Any, TypeVar
import warnings
from torch.nn.modules import Module
from einops import rearrange
from torch.optim.lr_scheduler import _LRScheduler
from utils.metric_utils import *
import argparse
from DataLoader import Data_transform
import models

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def set_wandb(args):
    wandb.init(project=args.project_name)
    wandb.config.update(args, allow_val_change=True)
    
    if args.train_type == 'cold':
        wandb.run.name = f"{args.optimizer_type.capitalize()}_Cold {str(args.seed)}"
    elif args.train_type == 'warm':
        wandb.run.name = f"{args.optimizer_type.capitalize()}_Warm {str(args.seed)}" 
    elif args.train_type == 'warm_rm':
        wandb.run.name = f"{args.optimizer_type.capitalize()}_WarmReM {str(args.seed)}"  
    elif args.train_type == 'sp':
        wandb.run.name = f"{args.optimizer_type.capitalize()}+S&P({args.sp_lambda}) {str(args.seed)}"
    elif args.train_type == 'dash':
        wandb.run.name = f"{args.optimizer_type.capitalize()}+DASH({args.dash_lambda}/{args.dash_alpha}) {str(args.seed)}"

    elif args.train_type == 'l2_init':
        wandb.run.name = f"{args.optimizer_type.capitalize()}+L2_INIT({args.l2_init_lambda}) {str(args.seed)}"
    elif args.train_type == 'reset':
        wandb.run.name = f"{args.optimizer_type.capitalize()}+Reset {str(args.seed)}"

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

def l2_init(args, current_model, init_model, loss):
    if (args.train_type == 'l2_init') & (init_model is not None):
        reg = 0
        for m1, m2 in zip(current_model.modules(), init_model.modules()):
            if hasattr(m1, 'weight'):
                if m1.weight is not None:
                    reg += (m1.weight - m2.weight.detach()).norm()
        reg = args.l2_init_lambda * (reg ** 2)
        loss += reg
    return loss
    
def return_loss(args, criterion, outputs, targets, current_model, init_model):
    loss = criterion(outputs, targets)
    loss = l2_init(args, current_model, init_model, loss)
    return loss

def get_optimizer(args, model):
    if args.optimizer_type == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            momentum=args.momentum
        )
    elif args.optimizer_type == 'sam':
        optimizer = SAM(
            model.parameters(), 
            optim.SGD, 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay, 
            momentum=args.momentum,
            rho=args.rho
        )
    else:
        raise NotImplemented
        
    return optimizer

def dash(args, model, criterion, chunk_loader):
    ####################
    #### initialize
    temp_net = copy.deepcopy(model)
    all_grad_lst = []
    all_grad_bias_lst = []
    
    for m in temp_net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            all_grad_lst.append(torch.zeros_like(m.weight))
            if m.bias != None:
                all_grad_bias_lst.append(torch.zeros_like(m.bias))
                
    ####################
    #### Gather Gradient
    for i, (x_batch, y_batch) in enumerate(chunk_loader):
        temp_net.zero_grad()
        y_hat = temp_net(x_batch.to(args.device))
        loss = criterion(y_hat, y_batch.to(args.device))
        loss.backward()

        step = 0
        bias_step = 0
        for m in temp_net.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                all_grad_lst[step] = (1 - args.dash_alpha) * all_grad_lst[step] + args.dash_alpha * (-m.weight.grad.detach())
                if m.bias != None:
                    all_grad_bias_lst[bias_step] = (1 - args.dash_alpha) * all_grad_bias_lst[bias_step] + args.dash_alpha * (-m.bias.grad.detach()) 
                    bias_step += 1
                step += 1
        torch.cuda.empty_cache()
    #########################
    ## Shrink
    step = 0
    bias_step = 0
    
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                grad = rearrange(all_grad_lst[step], 'o i h w -> o i (h w)')
                param_r = rearrange(m.weight, 'o i h w -> o i (h w)')
                cos_sim = torch.cosine_similarity(grad, param_r, dim=-1)
                scale = torch.clamp(cos_sim, min=args.dash_lambda, max=1)
                param_r.mul_(scale[:, :, None])
                step += 1
                
                if m.bias is not None:
                    cos_sim_bias = torch.cosine_similarity(all_grad_bias_lst[bias_step].reshape(1, -1), m.bias.reshape(1, -1))
                    scale_bias = torch.clamp(cos_sim_bias, min=args.dash_lambda, max=1)
                    m.bias.mul_(scale_bias)
                    bias_step += 1
                
            elif isinstance(m, nn.Linear):
                cos_sim = torch.cosine_similarity(all_grad_lst[step], m.weight, dim=-1)
                scale = torch.clamp(cos_sim, min=args.dash_lambda, max=1)
                m.weight.mul_(scale[:, None])
                step += 1
                if m.bias is not None:
                    cos_sim_bias = torch.cosine_similarity(all_grad_bias_lst[bias_step].reshape(1, -1), m.bias.reshape(1, -1))
                    scale_bias = torch.clamp(cos_sim_bias, min=args.dash_lambda, max=1)
                    m.bias.mul_(scale_bias)
                    bias_step += 1

    return model

def perform_dash(args, model, buffer, criterion, num_iters_per_chunk):
    if args.sota == True:
        dash_buffer = copy.deepcopy(buffer)
        for data in dash_buffer:
            data.dataset.transform = None
        dash_data = Data_transform(ConcatDataset(dash_buffer), transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(args.mean, args.std)]))
    else:
        dash_data = ConcatDataset(buffer)
    recent_chunk_loader = DataLoader(dash_data, batch_size=num_iters_per_chunk, shuffle=False)
    model = dash(args, model, criterion, recent_chunk_loader)

    return model

def post_training_actions(args, model, optimizer, chunk_idx):
    if args.train_type == 'cold':
        model = models.get_model(args).to(args.device)
        optimizer = get_optimizer(args, model)

    elif args.train_type == 'sp':
        model.weight_sp(args)

    elif args.train_type == 'reset':
        model.fc_reset(args, chunk_idx+1)

    elif args.train_type == 'warm_rm':
        optimizer = get_optimizer(args, model)

    return model, optimizer

## From ConvNeXt
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x