import torch
import torch.nn.functional as F
import numpy as np
import copy

def add_normal_noise_to_params(model, scale):
    """Add normal noise to model parameters"""
    noise = {}
    for name, param in model.named_parameters():
        noise[name] = torch.normal(mean=0, std=scale, size=param.size()).to(param.device)
        param.data.add_(noise[name])
    return noise

def add_uniform_noise_to_params(model, scale, original_weights):
    """Add uniform noise scaled by original weights to model parameters"""
    noise = {}
    for name, param in model.named_parameters():
        noise[name] = torch.empty_like(param).uniform_(-scale/2, scale/2)
        noise[name] *= original_weights[name]
        param.data.add_(noise[name])
    return noise


def compute_accuracy(model, data_loader):
    """Compute the accuracy of the model"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    model.train()
    return accuracy


def apply_gradient_ascent(model, optimizer, images, labels, ascent_step):
    """Perform gradient ascent to maximize the loss"""
    for _ in range(ascent_step):
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        (-loss).backward()
        optimizer.step()


def clip_params(model, original_weights, magnitude):
    """Clip parameters to stay within magnitude-aware bounds."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            max_perturb = (torch.abs(original_weights[name]) + 1) * magnitude
            upper_bound = original_weights[name] + max_perturb
            lower_bound = original_weights[name] - max_perturb
            param.data.clamp_(lower_bound, upper_bound)



def sharpness_sigma(
    model, data_loader, training_accuracy, target_deviate, upper=5., lower=0., search_depth=20, mtc_iter=15,
    ascent_step=20, deviat_eps=1e-2, bound_eps=5e-3):
    """Finds the maximum standard deviation on the weights that does not impact the training accuracy by more than `target_deviate`"""

    # Load model checkpoint
    

    # Create a deep copy of the original model
    original_model = copy.deepcopy(model)
    
    model.train()
    

    # Set optimizer for gradient ascent
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    h, l = upper, lower
    for j in range(search_depth):
        m = (h + l) / 2.
        print(f'current sigma: {m}. n mtc: {mtc_iter}')
        min_accuracy = 1.0
        estimates = []
        
        for i in range(mtc_iter):
            model = copy.deepcopy(original_model)
            model.train()

            add_normal_noise_to_params(model, scale=m)
            
            for images, labels in data_loader:
                apply_gradient_ascent(model, optimizer, images, labels, ascent_step)
            
            accuracy = compute_accuracy(model, data_loader)
            estimates.append(accuracy)
            min_accuracy = min(min_accuracy, np.mean(estimates))

            if i % 2 == 0:
                print(f'mean: {np.mean(estimates)}  var: {np.var(estimates)}')
        
        deviate = abs(min_accuracy - training_accuracy)
        if h - l < bound_eps or abs(deviate - target_deviate) < deviat_eps:
            return m
        if deviate > target_deviate:
            h = m
        else:
            l = m

    return (h + l) / 2.