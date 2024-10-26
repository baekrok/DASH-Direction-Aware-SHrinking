import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, random_split, DataLoader
import torchvision
import torchvision.transforms as transforms
from einops import rearrange
from dotmap import DotMap
import argparse
import models
import os
import tqdm
import random
import numpy as np
import wandb
import copy
from sklearn.model_selection import train_test_split
from utils.SAM import SAM
from utils.metric_utils import *
from utils.train_utils import *
from DataLoader import *

def run(args):
    ########################
    ## Setup Configuration
    args = DotMap(args)
    args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    set_wandb(args)
    
    ########################
    ## Prepare Dataset
    trainset, _, testset, test_loader = get_data(args)
    prev_chunk_loader = None
    
    ########################
    ## Prepare Model, Criterion, Optimizer
    model = models.get_model(args).to(args.device)
    init_model = copy.deepcopy(model) # To use when l2_init
    
    # criterion & optimizer
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = get_optimizer(args, model)
    
    # Setup
    num_iters_per_chunk = len(trainset) // args.num_chunks
    chunks = random_split(trainset, [num_iters_per_chunk] * args.num_chunks)
    buffer = []
    
    # training loop
    test_acc_list, num_step_list = [], []

    for chunk_idx in range(args.num_chunks):
        buffer.append(chunks[chunk_idx])
        
        # Get dataloader for train
        chunk_loader = DataLoader(ConcatDataset(buffer), 
                                  batch_size=args.batch_size, 
                                  shuffle=True, 
                                  num_workers=args.num_workers)
        # For metric, not for train
        current_chunk_loader = DataLoader(chunks[chunk_idx], 
                                          batch_size=args.batch_size, 
                                          shuffle=True)
        chunk_loader_lst = [prev_chunk_loader, chunk_loader, current_chunk_loader]
        
        # DASH
        if (args.train_type == 'dash') & (chunk_idx >= 1):
            model = perform_dash(args, model, buffer, criterion, num_iters_per_chunk)
        
        # Get LR scheduler if SoTA setting
        if args.sota == True:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=0.2)
            warmup_scheduler = WarmUpLR(optimizer, len(chunk_loader))
            schedulers = [scheduler, warmup_scheduler]
        else:
            schedulers = [None, None]

        # Train
        train_acc, train_loss, step, epoch = train_chunk(args, model, init_model, criterion, optimizer, schedulers, chunk_loader_lst, chunk_idx)
        
        # Evaluate
        logs = evaluate(args, test_loader, model, train_acc, train_loss, step, epoch)
        test_acc_list, num_step_list, logs = update_metrics(test_acc_list, num_step_list, logs)
        
        wandb.log(logs, step=chunk_idx+1)
        
        # Post-training actions (Depends on train_type)
        model, optimizer = post_training_actions(args, model, optimizer, chunk_idx)
        prev_chunk_loader = chunk_loader # For metric
        torch.cuda.empty_cache()

def train_chunk(args, model, init_model, criterion, optimizer, schedulers, chunk_loader_lst, chunk_idx):
    model.train()
    epoch, step, train_acc, train_loss = 0, 0, 0, 0
    _, chunk_loader, _ = chunk_loader_lst
    while True:
        # For Metric 
        if chunk_idx >= 1:
            prev_log = get_prev_metric(args, epoch, chunk_idx, model, 
                                       chunk_loader_lst)
        # Train one epoch
        epoch_acc, epoch_loss = train_epoch(args, model, init_model, criterion, optimizer, schedulers[1], chunk_loader, epoch)
        
        if args.sota == True:
            schedulers[0].step()
 
        epoch += 1
        step += len(chunk_loader)
        train_acc += epoch_acc
        train_loss += epoch_loss
        
        if (epoch_acc >= args.converge_acc) or (epoch >= args.max_epoch):
            break
    
    return train_acc/epoch, train_loss/epoch, step, epoch

def train_epoch(args, model, init_model, criterion, optimizer, warmup_scheduler, chunk_loader, epoch):
    total_acc, total_loss = 0, 0
    for inputs, targets in chunk_loader:
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = return_loss(args, criterion, outputs, targets, model, init_model)
        loss.backward()
        
        if args.optimizer_type == 'sgd':
            optimizer.step()
        elif args.optimizer_type == 'sam':
            optimizer.first_step(zero_grad=True)
            return_loss(args, criterion, model(inputs), targets, model, init_model).backward()
            optimizer.second_step(zero_grad=True)
        if (args.sota == True) & (epoch < 1):
            warmup_scheduler.step()

        total_acc += (outputs.argmax(1) == targets).float().mean().item()
        total_loss += loss.item()
        
    return total_acc / len(chunk_loader), total_loss / len(chunk_loader)

if __name__=='__main__':
    # basic config
    parser = argparse.ArgumentParser()
    # For MLP
    parser.add_argument('--width',            type=int, default=1000)
    parser.add_argument('--depth',            type=int, default=2)
    ##############
    parser.add_argument('--converge_acc',     type=float, default=0.999)
    parser.add_argument('--train_type',       type=str, default='warm', 
                        choices=['warm', 'warm_rm', 'cold', 'sp', 'dash', 'l2_init', 'reset'])
    parser.add_argument('--model',            type=str, default='resnet18', choices=models.get_available_models())
    parser.add_argument('--dataset',          type=str, default='cifar10', choices=['cifar10', 'imagenet', 'cifar100', 'fashion_mnist', 'svhn'])
    parser.add_argument('--optimizer_type',   type=str, default='sgd', choices=['sgd', 'sam'])
    parser.add_argument('--rho',              type=float, default=0.1)
    parser.add_argument('--norm',             type=str, default='bn')
    parser.add_argument('--weight_decay',     type=float, default=0)
    parser.add_argument('--batch_size',       type=int, default=128)
    parser.add_argument('--test_batch_size',  type=int, default=256)
    parser.add_argument('--learning_rate',    type=float, default=0.001)
    parser.add_argument('--momentum',         type=float, default=0.9)
    parser.add_argument('--gpu',              type=int, default=0)
    parser.add_argument('--num_chunks',       type=int, default=50)
    parser.add_argument('--num_workers',      type=int, default=0)
    parser.add_argument('--seed',             type=int, default=2021)
    parser.add_argument('--sp_lambda',        type=float, default=0.3)
    parser.add_argument('--dash_lambda',      type=float, default=0.3)
    parser.add_argument('--dash_alpha',       type=float, default=0.3)
    parser.add_argument('--l2_init_lambda',   type=float, default=1e-2)
    parser.add_argument('--max_epoch',        type=int,   default=10000)
    parser.add_argument('--sota',             type=str2bool, default=False)
    args = parser.parse_args()
    
    # set project name
    if 'mlp' in args.model:
        args.project_name = f'{args.model}-{args.dataset}-d{args.depth}-w{args.width}-sota{args.sota}'
    else:
        args.project_name = f'{args.model}-{args.dataset}-sota{args.sota}'
        
    # set seeds
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    # run
    run(vars(args))
    
