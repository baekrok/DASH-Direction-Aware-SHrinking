import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, random_split
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
from torch.utils.data import Subset, DataLoader
from utils.SAM import SAM
from utils.metric_utils import *
from utils.train_utils import *
from DataLoader import *


def run(args):
    ########################
    ## Setup Configuration
    args = DotMap(args)
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    args.device = device
    set_wandb(args)
    
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    ########################
    ## Prepare Dataset
    trainset, _, testset, test_loader = get_data(args)
    
    ########################
    ## Prepare Model, Criterion, Optimizer
    model = models.get_model(args).to(device)
    init_model = copy.deepcopy(model) # To use when l2_init
    
    # criterion & optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = get_optimizer(args, model)
    
    ###################
    # Setup
    num_iters_per_chunk = len(trainset) // args.num_chunks
    chunks = random_split(trainset, [num_iters_per_chunk] * args.num_chunks)
    buffer = []
    
    ##################
    # training loop
    test_acc_list, num_step_list = [], []

    for chunk_idx in range(args.num_chunks):
        buffer.append(chunks[chunk_idx])
        
        meter_set = AverageMeterSet()
        grad_meter = AverageMeterSet()
        
        # Get dataloader
        chunk_loader = DataLoader(ConcatDataset(buffer), 
                                  batch_size=args.batch_size, 
                                  shuffle=True, 
                                  num_workers=args.num_workers)
        # Get LR scheduler if SoTA setting
        if args.sota == True:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=0.2)
            warmup_scheduler = WarmUpLR(optimizer, len(chunk_loader))
        
        ######################
        ## DASH 
        if (args.train_type == 'dash') & (chunk_idx >= 1):
            model = perform_dash(args, model, buffer, num_iters_per_chunk)
        ########################
        ## Train 
        epoch, step = 0, 0
        prev_acc_log = {}
        
        while True:
            ######################
            ## For Metric 
            # Previous Accuracy
            if (epoch <= 5) & (chunk_idx >= 1):
                prev_acc = get_prevacc(args, model, prev_chunk_loader)
                prev_acc_log[f'metrics/epoch{epoch}_prev_acc'] = prev_acc
            
            # Grad Norm in Epoch 0
            current_chunk_loader = DataLoader(chunks[chunk_idx], batch_size=args.batch_size, shuffle=True)
            if (epoch == 0) & (chunk_idx >= 1):
                for data_loader, key in zip([prev_chunk_loader, chunk_loader, current_chunk_loader],
                                            ['prev', 'all', 'current']):
                    prev_acc_log[f'metrics/epoch0_{key}_gradnorm'] = get_gradnorm(args, model, data_loader, criterion)
            
            #####################
            ## Train Model
            model.train()
            train_acc = 0
            for i, (inputs, targets) in enumerate(chunk_loader):
                inputs = inputs.to(args.device)
                targets = targets.to(args.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = outputs.max(1)

                loss = return_loss(args, criterion, outputs, targets, model, init_model)
                loss.backward()
                if args.optimizer_type == 'sgd':
                    optimizer.step()

                elif args.optimizer_type == 'sam':
                    optimizer.first_step(zero_grad=True)
                    outputs = model(inputs)
                    second_loss = return_loss(args, criterion, outputs, targets, model, init_model)
                    second_loss.backward()
                    optimizer.second_step(zero_grad=True)
                    
                if args.sota == True:
                    if epoch < 1:
                        warmup_scheduler.step()
                        
                acc = preds.eq(targets).float().mean()
                step += 1
                train_acc += acc.item()
                meter_set.update('train_metrics/train_loss', loss.item())
                meter_set.update('train_metrics/train_acc', train_acc / (i+1))
                
            epoch += 1
            train_acc /= i+1
            
            if args.sota == True:
                scheduler.step()
            # break if converge
            if (train_acc >= args.converge_acc) | (epoch >= args.max_epoch):
                break
                
        ##############################
        # Evaluation
        logs = {}
        logs.update(prev_acc_log)
        logs.update(meter_set.averages())
        logs.update(get_test_acc(args, test_loader, model, device))
        
        test_acc_list.append(logs['train_metrics/test_acc'])
        num_step_list.append(step)
        
        logs['train_metrics/test_acc_avg'] = np.mean(test_acc_list)
        logs['train_metrics/train_acc_at_last_epoch'] = train_acc
        logs['train_metrics/num_epochs'] = epoch
        logs['train_metrics/num_steps'] = step
        logs['train_metrics/num_steps_avg'] = np.mean(num_step_list)
        
        wandb.log(logs, step=chunk_idx+1)

        if args.train_type == 'cold':
            model = models.get_model(args).to(device)
            optimizer = get_optimizer(args, model)
            
        elif args.train_type == 'sp':
            model.weight_sp(args)

        elif args.train_type == 'reset':
            model.fc_reset(args, chunk_idx+1)
            
        elif args.train_type == 'warm_rm':
            optimizer = get_optimizer(args, model)
            
        prev_chunk_loader = chunk_loader
        torch.cuda.empty_cache()
        

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
        
    run(vars(args))