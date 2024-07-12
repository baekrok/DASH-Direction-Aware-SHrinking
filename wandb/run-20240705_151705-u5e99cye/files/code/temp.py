import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
import wandb
import models
from utils.train_utils import get_optimizer
import argparse
from dotmap import DotMap
from utils.metric_utils import *
from utils.train_utils import *
from DataLoader import *
import tqdm
import numpy as np

# Create custom dataset with experiment labels
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __getitem__(self, index):
        img, _ = self.dataset[index]
        return img, self.labels[index]

    def __len__(self):
        return len(self.dataset)

            
def run_online(args):
    args = DotMap(args)
    wandb.init(project=args.project_name)
    wandb.config.update(args)

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    args.device = device

    # Prepare dataset
    trainset, _, testset, test_loader = get_data(args)
    data_length = len(trainset)
    chunk_size = 25000
    num_chunks = data_length // chunk_size

    # Create random labels
    random_labels = torch.randint(0, 10, (data_length,))

    # Create chunks
    chunk_indices = [list(range(i*chunk_size, (i+1)*chunk_size)) for i in range(num_chunks)]

    # Prepare model
    model = models.get_model(args).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
        
    for experiment in range(num_chunks):
        print(f"Experiment {experiment + 1}/{num_chunks}")

        # Create labels for this experiment
        experiment_labels = random_labels.clone()
        for i in range(experiment + 1):
            experiment_labels[chunk_indices[i]] = torch.tensor([trainset[j][1] for j in chunk_indices[i]])

        custom_trainset = CustomDataset(trainset, experiment_labels)
        train_loader = DataLoader(custom_trainset, batch_size=args.batch_size, shuffle=True)

        optimizer = get_optimizer(args, model)

        for epoch in tqdm.tqdm(range(args.epoch)):
            model.train()
            train_loss = 0
            train_acc = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                _, preds = outputs.max(1)
                train_loss += loss.item()
                train_acc += preds.eq(targets).float().mean().item()
                
            train_loss /= len(train_loader)
            train_acc /= len(train_loader)
            
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
                'epoch': epoch + experiment * args.epoch,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'experiment': experiment,
                'correct_chunks': experiment + 1
            })
            if train_acc >= 0.99:
                break
                
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch',            type=int, default=10)
    parser.add_argument('--model',            type=str, default='vgg19', choices=models.get_available_models())
    parser.add_argument('--dataset',          type=str, default='cifar10')
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
    args.project_name = 'cifar10_random_to_correct'
    run_online(vars(args))