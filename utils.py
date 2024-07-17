import os
from time import sleep
import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, SequentialSampler
import importlib
import copy
import argparse
# import measures
# from measures import calculate, get_measures, get_bounds, compute_kendall
from torchvision import transforms, datasets
import json
import matplotlib.pyplot as plt
import csv
import signal
import hashlib

# train the model for one epoch on the given set
def train(args, model, device, train_loader, criterion, optimizer, epoch, batches):
    sum_loss, sum_correct = 0, 0
    n_layers = len([p for name, p in model.named_parameters() if p.requires_grad and 'weight' in name])
    gradients = {}
    model.train()

    for i, (data, target) in enumerate(train_loader):
        if i >= batches:
            break
        # print(f'Batch {i+1}/{batches}', end='\r')
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        # Zero out the gradients
        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)
        
        pred = output.max(1)[1]
        sum_correct += pred.eq(target).sum().item()
        sum_loss += len(data) * loss.item()
        # compute the gradient
        loss.backward()


        optimizer.step()
    return sum_correct / len(train_loader.dataset), 1 - (sum_correct / len(train_loader.dataset)), sum_loss / len(train_loader.dataset)

# evaluate the model on the given set
def validate(args, model, device, val_loader, criterion, batches = -1):
    sum_loss, sum_correct = 0, 0
    margin = torch.Tensor([]).to(device)

    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            if batches > 0 and i >= batches:
                break
            # print(f'Batch {i+1}/{batches}', end='\r')
            data, target = data.to(device), target.to(device)

            output = model(data)

            pred = output.max(1)[1]
            sum_correct += pred.eq(target).sum().item()
            sum_loss += len(data) * criterion(output, target).item()

            # compute the margin
            output_m = output.clone()
            for i in range(target.size(0)):
                output_m[i, target[i]] = output_m[i,:].min()
            margin = torch.cat((margin, output[:, target].diag() - output_m[:, output_m.max(1)[1]].diag()), 0)
        val_margin = np.percentile( margin.cpu().numpy(), 10 )

    return sum_correct / len(val_loader.dataset), 1 - (sum_correct / len(val_loader.dataset)), sum_loss / len(val_loader.dataset), val_margin**2

def compute_hash(data):
    return hashlib.sha256(data.cpu().numpy().tobytes()).hexdigest()

def get_targets(device, data_loader, n_images):
    y = torch.randn((n_images, 1)).to(device)
    for i, (_, target) in enumerate(data_loader):
        if i >= n_images:
            break
        y[i] = target.to(device)
    return y


def ntk(grad_x, grad_y, degree = 1, c = 0):
    return torch.pow(torch.add(torch.dot(grad_x, grad_y), c), degree).item()

def compute_graam(model, device, data_loader, n_images, criterion, kernel_f = ntk, approximation = False, testing = False):
    if approximation:
        return compute_graam_approximation(model, device, data_loader, n_images, criterion, kernel_f)
    
    if testing:
        return compute_graam_testing(model, device, data_loader, n_images, criterion, kernel_f)
    return compute_graam_unoptimized(model, device, data_loader, n_images, criterion, kernel_f)


def compute_graam_approximation(model, device, data_loader, n_images, criterion, kernel_f = ntk):
    pass


def compute_graam_testing(model, device, data_loader: DataLoader, n_images, criterion, kernel_f = ntk):
    model.eval()

    kernel_matrix = torch.zeros(n_images, n_images).to(device)

    data_hashes = []

    for i, (data, target) in enumerate(data_loader):
        if i >= n_images:
            break
        # print(f'Batch {i+1}/{batches}', end='\r')
        data, target = data.to(device), target.to(device)
        data_hash = compute_hash(data)
        data_hashes.append(data_hash)
    
    for i, data_hash in enumerate(data_hashes):
        print(f'image {i+1}: {data_hash}')

    return kernel_matrix

def compute_graam_unoptimized(model, device, data_loader, n_images, criterion, kernel_f = ntk):
    # Computes the Kernel Matrix, usually with regards to the Neural Tangent Kernel (NTK(x, x') = <grad f(x), grad f(x')>)

    model.eval()

    kernel_matrix = torch.zeros(n_images, n_images).to(device)

    grads = []
    # Since the Kernel matrix is symmetric, we only compute the lower triangular part
    # Idea: given the current batch, we compute it's gradient
    for i, (data, target) in enumerate(data_loader):
        if i >= n_images:
            break
        # print(f'Batch {i+1}/{batches}', end='\r')
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output = model(data)
        loss = criterion(output, target)

        # compute the gradient
        model.zero_grad()
        loss.backward()

        # Get the gradients from model.named_parameters in a Tensor
        current_grad = torch.cat([p.grad.view(-1) for name, p in model.named_parameters() if p.grad is not None and 'weight' in name]).view(-1).to(device)

        grads.append(current_grad)
        kernel_matrix[i, i] = kernel_f(current_grad, current_grad)
    
        # Compute for j < i
        for j in range(i):
            kernel_matrix[i, j] = kernel_f(current_grad, grads[j])
            kernel_matrix[j, i] = kernel_matrix[i, j] # Symmetric
        
    return kernel_matrix


def save_model(model, optimizer):
    model_save = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    return model_save


def save_models(model_saves, path):
    for i, model in enumerate(model_saves):
        if i == 0:
            torch.save(model, f'{path}/init.pth')
            continue
        torch.save(model, f'{path}/epoch_{i}.pth')


def load_parameter(model, filename):
    model.load_state_dict(torch.load(filename))
    return model


def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    epoch = checkpoint['epoch']
    args = checkpoint['args']

    # Delete the checkpoint file
    os.remove(filename)
    return model, optimizer, epoch, args


def get_all_data(data_loader):
    all_data, all_target = [], []
    for data, target in data_loader:
        all_data.append(data)
        all_target.append(target)
    print(type(all_data))
    print(len(all_data))
    print(all_data[0])
    




# Load and Preprocess data.
def load_data(split, dataset_name, datadir, nchannels):

    if dataset_name == 'MNIST':
        normalize = transforms.Normalize(mean=[0.131], std=[0.289])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    tr_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        normalize,
        ])
    
    val_transform = transforms.Compose([

        transforms.Resize(32),
        transforms.ToTensor(),
        normalize
        ])
    
    # torchvision.datasets.dataset_name
    get_dataset = getattr(datasets, dataset_name)
    # print(datadir)
    # The class call varies slightly from one dataset to another. Refer to torchvision.datasets doc of your version
    if dataset_name == 'SVHN':
        if split == 'train':
            dataset = get_dataset(root=datadir, split='train', download=True, transform=tr_transform)
        else:
            dataset = get_dataset(root=datadir, split='test', download=True, transform=val_transform)
    else:
        if split == 'train':
            dataset = get_dataset(root=datadir, train=True, download=True, transform=tr_transform)
        else:
            dataset = get_dataset(root=datadir, train=False, download=True, transform=val_transform)

    return dataset


# load the parameter for the neural network
# if greater than 1, then repeat the value n_nn times
# else load the value from the file
# consist of a single row csv file,
def load_parameter(name, arg, n_nn, type_val=float):
    if arg > -1:
        return [arg] * n_nn
    else:
        with open(f'csv/{name}.csv', 'r') as f:
            reader = csv.reader(f)
            row = next(reader)
            return [type_val(x) for x in row[:n_nn]]
