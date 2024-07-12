import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
import wandb
import copy
import models
from utils.train_utils import get_optimizer
from resnet import resnet18
import argparse
from dotmap import DotMap
from utils.metric_utils import *
from utils.train_utils import *
from DataLoader import *
import tqdm

def run_online(args):
    args = DotMap(args)
    wandb.init(project=args.project_name)
    wandb.config.update(args)

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    args.device = device

    # Prepare dataset
    trainset, _, testset, test_loader = get_data(args)
    data_length = len(trainset)
    chunks = random_split(trainset, [data_length // 2, data_length // 2])

    # Prepare model
    model = models.get_model(args).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = get_optimizer(args, model)
    
    for chunk_idx in range(args.num_chunks):
        buffer = chunks[chunk_idx]
        chunk_loader = DataLoader(buffer, batch_size=args.batch_size, shuffle=True)
        
        if args.train_type in ['random_label', 'random_label_rm']:
            if chunk_idx == 0:
                _, chunk_loader, _, _ = get_data(args, get_random=True)
            else:
                _, chunk_loader, _, _ = get_data(args, get_random=False)
        
        for epoch in tqdm.tqdm(range(args.epoch)):
            model.train()
            train_loss = 0
            train_acc = 0
            for inputs, targets in chunk_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                _, preds = outputs.max(1)
                train_loss += loss.item()
                train_acc += preds.eq(targets).float().mean().item()
            
            train_loss /= len(chunk_loader)
            train_acc /= len(chunk_loader)

            # Test
            model.eval()
            test_loss = 0
            test_acc = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    _, preds = outputs.max(1)
                    test_loss += loss.item()
                    test_acc += preds.eq(targets).float().mean().item()
            
            test_loss /= len(test_loader)
            test_acc /= len(test_loader)

            # Log to wandb
            wandb.log({
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'epoch': epoch + chunk_idx * args.epoch,
                'train_loss': train_loss,
                'test_loss': test_loss
            })

        if chunk_idx == 0:
            # Reset or reinitialize model for the second chunk if needed
            if args.train_type == 'random':
                model = models.get_model(args).to(device)
                optimizer = get_optimizer(args, model)
            elif args.train_type == 'warm_rm':
                optimizer = get_optimizer(args, model)
            elif args.train_type == 'random_label_rm':
                optimizer = get_optimizer(args, model)
    wandb.finish()

if __name__ == '__main__':
    # (Keep the existing argument parsing code)
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch',            type=int, default=500)
    parser.add_argument('--train_type',       type=str, default='warm', 
                        choices=['warm', 'warm_rm', 'random', 'random_label', 'random_label_rm'])
    parser.add_argument('--model',            type=str, default='vgg19', choices=models.get_available_models())
    parser.add_argument('--dataset',          type=str, default='cifar10', choices=['cifar10', 'imagenet', 'cifar10_sota', 'imagenet_sota', 'cifar100', 'cifar100_sota', 'fashion_mnist', 'svhn', 'cifar10_sota_woaug'])
    parser.add_argument('--optimizer_type',   type=str, default='sgd', choices=['sgd', 'sam'])
    parser.add_argument('--norm',             type=str, default='bn')
    parser.add_argument('--weight_decay',     type=float, default=0)
    parser.add_argument('--batch_size',       type=int, default=128)
    parser.add_argument('--test_batch_size',  type=int, default=1024)    
    parser.add_argument('--learning_rate',    type=float, default=0.001)
    parser.add_argument('--momentum',         type=float, default=0.9)
    parser.add_argument('--activation',       type=str,  default='ReLU')
    parser.add_argument('--gpu',              type=int, default=0)
    parser.add_argument('--seed',             type=int, default=2021)
    
    args = parser.parse_args()
    args.num_chunks = 2  # Force num_chunks to 2
    args.project_name = 'noise vs warm'
    run_online(vars(args))