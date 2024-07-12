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

    wandb.init(project=args.project_name)
    wandb.config.update(args, allow_val_change=True)
    
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    args.device = device
    
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

    num_iters_per_chunk = len(trainset) // args.num_chunks
    chunks = random_split(trainset, [num_iters_per_chunk] * args.num_chunks)
    buffer = []
    args.prev_epoch = 0
    
    ##################
    # training loop
    test_acc_list = []
    num_step_list = []

    for chunk_idx in tqdm.tqdm(range(args.num_chunks)):
        buffer.append(chunks[chunk_idx])
        model.train()
        
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
            if args.sota == True:
                dash_buffer = copy.deepcopy(buffer)
                for data in dash_buffer:
                    data.dataset.transform = None
                dash_data = Data_transform(ConcatDataset(es_buffer), transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize(args.mean, args.std)]))
            else:
                dash_data = ConcatDataset(buffer)
            recent_chunk_loader = DataLoader(dash_data, batch_size=int(num_iters_per_chunk), shuffle=False)

            model = dash(args, model, criterion, recent_chunk_loader)
            
        ########################
        ## Train 
        done_with_chunk = 0
        epoch = 0
        step = 0
        
        prev_acc_log = {}
        while not done_with_chunk:
            ############ For Metric ###############################################################################
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
            
            #######################################################################################################
            # Train Model
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
                meter_set.update('train_metrics/train_acc', train_acc)
                
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
        
def get_test_acc(args, loader, model, device):    
    meter_set = AverageMeterSet()
    model.eval()

    # evaluation loop
    criterion = nn.CrossEntropyLoss().to(device)
    for inputs, targets in loader:
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)
        
        outputs = model(inputs)
        _, preds = outputs.max(1)
        loss = criterion(outputs, targets)
        acc = preds.eq(targets).float().mean()
        
        logs = {
            'train_metrics/test_loss': loss.item(),
            'train_metrics/test_acc': acc.item(),
            }
        
        for key, value in logs.items():
            meter_set.update(key, value)
            
    test_logs = meter_set.averages()
    return test_logs


""" # For Additional Metrics
def test_online(args, loader, model, prev_model, prev_model_sh, init_model, train_batches, device):
    meter_set = AverageMeterSet()
    model.eval()

    # evaluation loop
    criterion = nn.CrossEntropyLoss().to(device)
    for inputs, targets in loader:
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)
        
        outputs = model(inputs)
        _, preds = outputs.max(1)
        loss = criterion(outputs, targets)
        acc = preds.eq(targets).float().mean()
        
        # fisher trace
        fisher_trace = 0
        
        for key, param in model.named_parameters():
            grads = torch.autograd.grad(
                outputs=loss, inputs=param,
                create_graph=True, retain_graph=True, allow_unused=True
            )[0]
            # noisy linear's sigma will not be appended to fisher penalty
            # to inlcude those terms, set epsilon terms into requires_grad=True
            if grads is not None:
                fisher_trace += (grads ** 2).mean()
            torch.cuda.empty_cache()

        logs = {
            'train_metrics/fisher_trace': fisher_trace.item(),
            'train_metrics/test_loss': loss.item(),
            'train_metrics/test_acc': acc.item(),
            }
        
        for key, value in logs.items():
            meter_set.update(key, value)
            
    test_logs = meter_set.averages()
    if ('mlp' in args.model) | (args.dataset == 'cifar100_sota') | (args.dataset == 'cifar10_sota') | ('mnist' in args.dataset):
        test_met_logs = {'metrics/model_dist (From prev)': get_model_distance(args, prev_model, model),
                         'metrics/model_dist (From shrinked prev)': get_model_distance(args, prev_model_sh, model),
                         'metrics/model_dist (From init)': get_model_distance(args, init_model, model),
                         'metrics/AVG Weight Magnitude of fc after converge': get_fc_param_norm(args, model),
                        }
        torch.cuda.empty_cache()
    else:
        dead_neuron, feature_sing_vals, avg_sparsities, ns_active_relus_0p, ns_active_relus_1p, ns_active_relus_5p, ns_active_relus_10p = [], [], [], [], [], [], []


        feature_sing_vals = compute_feature_sing_vals(train_batches, model, device=device)
        phi_list = compute_feature_matrix(train_batches, model, device=device)
        for phi in phi_list:
            relu_threshold = phi.max() / 20
            avg_sparsities += [(phi > relu_threshold).mean()]
            dead_neuron += [((phi > 0.0).sum(0) == 0.).sum() / phi.shape[1]]
            ns_active_relus_0p += [((phi > relu_threshold).sum(0) > phi.shape[0] * 0.0).sum() / phi.shape[1]]
            ns_active_relus_1p += [((phi > relu_threshold).sum(0) > phi.shape[0] * 0.01).sum() / phi.shape[1]]
            ns_active_relus_5p += [((phi > relu_threshold).sum(0) > phi.shape[0] * 0.05).sum() / phi.shape[1]]
            ns_active_relus_10p += [((phi > relu_threshold).sum(0) > phi.shape[0] * 0.1).sum() / phi.shape[1]]

        feature1, feature2, feature3, feature4, feature5 = [np.sum(np.cumsum(svals**2) <= np.sum(svals**2) * 0.99) + 1 for svals in feature_sing_vals]
        avg_sparsities1, avg_sparsities2, avg_sparsities3, avg_sparsities4, avg_sparsities5 = avg_sparsities
        act_0_1, act_0_2, act_0_3, act_0_4, act_0_5 = ns_active_relus_0p
        act_1_1, act_1_2, act_1_3, act_1_4, act_1_5 = ns_active_relus_1p
        act_2_1, act_2_2, act_2_3, act_2_4, act_2_5 = ns_active_relus_5p
        act_3_1, act_3_2, act_3_3, act_3_4, act_3_5 = ns_active_relus_10p
        dead_neuron1, dead_neuron2, dead_neuron3, dead_neuron4, dead_neuron5 = dead_neuron
        
        test_met_logs = {'metrics/model_dist (From prev)': get_model_distance(args, prev_model, model),
                         'metrics/model_dist (From shrinked prev)': get_model_distance(args, prev_model_sh, model),
                         'metrics/model_dist (From init)': get_model_distance(args, init_model, model),
                         'metrics/AVG Weight Magnitude of backbone after converge': get_cnn_param_norm(args, model),
                         'metrics/AVG Weight Magnitude of fc after converge': get_fc_param_norm(args, model),

                         'feature_rank/layer1': feature1,
                         'feature_rank/layer2': feature2,
                         'feature_rank/layer3': feature3,
                         'feature_rank/layer4': feature4,
                         'feature_rank/layer5': feature5,

                         'avg_sparsities/layer1': avg_sparsities1,
                         'avg_sparsities/layer2': avg_sparsities2,
                         'avg_sparsities/layer3': avg_sparsities3,
                         'avg_sparsities/layer4': avg_sparsities4,
                         'avg_sparsities/layer5': avg_sparsities5,

                         'ns_active_relus_0p/layer1': act_0_1,
                         'ns_active_relus_0p/layer2': act_0_2,
                         'ns_active_relus_0p/layer3': act_0_3,
                         'ns_active_relus_0p/layer4': act_0_4,
                         'ns_active_relus_0p/layer5': act_0_5,

                         'ns_active_relus_1p/layer1': act_1_1,
                         'ns_active_relus_1p/layer2': act_1_2,
                         'ns_active_relus_1p/layer3': act_1_3,
                         'ns_active_relus_1p/layer4': act_1_4,
                         'ns_active_relus_1p/layer5': act_1_5,

                         'ns_active_relus_5p/layer1': act_2_1,
                         'ns_active_relus_5p/layer2': act_2_2,
                         'ns_active_relus_5p/layer3': act_2_3,
                         'ns_active_relus_5p/layer4': act_2_4,
                         'ns_active_relus_5p/layer5': act_2_5,

                         'ns_active_relus_10p/layer1': act_3_1,
                         'ns_active_relus_10p/layer2': act_3_2,
                         'ns_active_relus_10p/layer3': act_3_3,
                         'ns_active_relus_10p/layer4': act_3_4,
                         'ns_active_relus_10p/layer5': act_3_5,  

                         'dead_neuron_ratio/layer1': dead_neuron1, 
                         'dead_neuron_ratio/layer2': dead_neuron2, 
                         'dead_neuron_ratio/layer3': dead_neuron3, 
                         'dead_neuron_ratio/layer4': dead_neuron4, 
                         'dead_neuron_ratio/layer5': dead_neuron5, }
        
        torch.cuda.empty_cache()
    return test_logs, test_met_logs
"""

if __name__=='__main__':
    # basic config
    parser = argparse.ArgumentParser()
    # For MLP
    parser.add_argument('--width',            type=int, default=1000)
    parser.add_argument('--depth',            type=int, default=2)
    ##############
    parser.add_argument('--converge_acc',     type=float, default=0.999)
    parser.add_argument('--train_type',       type=str, default='warm', 
                        choices=['warm', 'warm_rm', 'random', 'sp', 'dash', 'l2_init', 'reset'])
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
    parser.add_argument('--sp_lambda',        type=float, default=0.1)
    parser.add_argument('--dash_lambda',      type=float, default=0.3)
    parser.add_argument('--dash_alpha',       type=float, default=0.3)
    parser.add_argument('--l2_init_lambda',   type=float, default=1e-2)
    parser.add_argument('--max_epoch',        type=int,   default=10000)
    parser.add_argument('--sota',             type=str2bool, default=False)
    args = parser.parse_args()
    
    # set project name
    if 'mlp' in args.model:
        args.project_name = f'{args.model}-{args.dataset}-d{args.depth}-w{args.width}'
    else:
        args.project_name = f'{args.model}-{args.dataset}'
        
    run(vars(args))