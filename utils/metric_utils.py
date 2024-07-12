import torch
import torch.nn as nn
import numpy as np


### For additional metrics ###
def get_grad_norm_stats(args, model):
    grad_norm = []
    stats = {}
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if module.weight.grad is not None:
                grad_norm.append(module.weight.grad.detach().norm(2).item()**2)
    stats['mean_grad_norm'] = np.sqrt(np.sum(grad_norm))

    return stats

def get_cnn_param_norm(args, model):
    total_sum = 0
    total_num = 0
    
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            abs_weights = np.abs(module.weight.cpu().detach().numpy())
            total_sum += np.sum(abs_weights)
            total_num += abs_weights.shape[0] * abs_weights.shape[1]

    return total_sum / total_num

def get_fc_param_norm(args, model):
    total_sum = 0
    total_num = 0

    for module in model.modules():
        if isinstance(module, nn.Linear):
            abs_weights = np.abs(module.weight.cpu().detach().numpy())
            total_sum += np.sum(abs_weights)
            total_num += abs_weights.shape[0] * abs_weights.shape[1]

    return total_sum / total_num

def get_model_distance(args, model1, model2):
    dist_travel = []
    for module1, module2 in zip(model1.modules(), model2.modules()):
        if isinstance(module1, nn.Conv2d) or isinstance(module1, nn.Linear):
            dist_travel.append(((module1.weight.detach().cpu() - module2.weight.detach().cpu()).norm() / module1.weight.detach().cpu().norm()).numpy())
    return np.array(dist_travel).mean()


def compute_feature_sing_vals(batches, model, device):
    with torch.no_grad():
        phi_list = []
        features1_list = []
        features2_list = []
        features3_list = []
        features4_list = []
        features5_list = []
        for i, (X, y) in enumerate(batches):
            X, y = X.to(device), y.to(device)
            features = model(X, return_features=True)
            feature_1, feature_2, feature_3, feature_4, feature_5 = features
            features1_list.append(feature_1)
            features2_list.append(feature_2)
            features3_list.append(feature_3)
            features4_list.append(feature_4)
            features5_list.append(feature_5)
            
        sing_vals_list = []
        for phi_list in [features1_list, features2_list, features3_list, features4_list, features5_list]:
            phi = torch.cat(phi_list)
            phi = phi.reshape(phi.shape[0], np.prod(phi.shape[1:]))
            phi = phi - torch.mean(phi, axis=1, keepdims=True)
            feature_sing_vals = torch.linalg.svdvals(phi).cpu().numpy()
            sing_vals_list.append(feature_sing_vals)
            
    torch.cuda.empty_cache()
    return sing_vals_list
      
def compute_feature_matrix(batches, model, device, n_batches=-1):
    with torch.no_grad():
        features1_list = []
        features2_list = []
        features3_list = []
        features4_list = []
        features5_list = []
        
        for i, (X,  y) in enumerate(batches):
            if n_batches != -1 and i > n_batches:
                break
            X, y = X.to(device), y.to(device)
            features = model(X, return_features=True)
            feature_1, feature_2, feature_3, feature_4, feature_5 = [feature.cpu().numpy() for feature in features]
            features1_list.append(feature_1)
            features2_list.append(feature_2)
            features3_list.append(feature_3)
            features4_list.append(feature_4)
            features5_list.append(feature_5)
        
        phi_list = []
        for feature in [features1_list, features2_list, features3_list, features4_list, features5_list]:
            phi = np.vstack(feature)
            phi = phi.reshape(phi.shape[0], np.prod(phi.shape[1:]))
            phi_list.append(phi)
            
    torch.cuda.empty_cache()
    return phi_list


####
def get_prev_metric(args, epoch, chunk_idx, model, chunk_loader_lst):
    criterion = nn.CrossEntropyLoss().to(args.device)
    prev_chunk_loader, chunk_loader, current_chunk_loader = chunk_loader_lst
    
    log = {}
    # Previous Accuracy
    if (epoch <= 5) & (chunk_idx >= 1):
        prev_acc = get_prevacc(args, model, prev_chunk_loader)
        log[f'metrics/epoch{epoch}_prev_acc'] = prev_acc

    # Grad Norm in Epoch 0
    if (epoch == 0) & (chunk_idx >= 1):
        for data_loader, key in zip([prev_chunk_loader, chunk_loader, current_chunk_loader],
                                    ['prev', 'all', 'current']):
            log[f'metrics/epoch0_{key}_gradnorm'] = get_gradnorm(args, model, data_loader, criterion)
    return log

def get_gradnorm(args, model, chunk_loader, criterion):
    model.zero_grad()
    model.train()
    loss = 0
    for i, (inputs, targets) in enumerate(chunk_loader):
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
    grad_norm = get_grad_norm_stats(args, model)['mean_grad_norm']
    model.zero_grad()
    torch.cuda.empty_cache()
    return grad_norm

def get_prevacc(args, model, chunk_loader):
    model.eval()
    train_acc = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(chunk_loader):
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            acc = preds.eq(targets).float().mean()
            train_acc += acc.item()
    model.train()
    torch.cuda.empty_cache()
    return train_acc / (i+1)

def evaluate(args, loader, model, train_acc, train_loss, step, epoch):    
    acc, loss = 0, 0
    
    # evaluation loop
    model.eval()
    criterion = nn.CrossEntropyLoss().to(args.device)
    for inputs, targets in loader:
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)
        
        outputs = model(inputs)
        _, preds = outputs.max(1)
        loss += criterion(outputs, targets).item()
        acc += preds.eq(targets).float().mean().item()

    logs = {
        'train_metrics/test_loss': loss/len(loader),
        'train_metrics/test_acc': acc/len(loader),
        'train_metrics/train_loss': train_loss,
        'train_metrics/train_acc': train_acc,
        'train_metrics/num_steps': step,
        'train_metrics/num_epochs': epoch
        }
        
    return logs
    
def update_metrics(test_acc_list, num_step_list, logs):
    test_acc_list.append(logs['train_metrics/test_acc'])
    num_step_list.append(logs['train_metrics/num_steps'])
    
    logs.update({'train_metrics/test_acc_avg': np.mean(test_acc_list),
                 'train_metrics/num_steps_avg': np.mean(num_step_list)})
    
    return test_acc_list, num_step_list, logs