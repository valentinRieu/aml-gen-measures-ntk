from contextlib import contextmanager
from typing import List, Optional
import torch
import math
import copy
import warnings
import torch.nn as nn
from torch import Tensor
import pdb
import numpy as np
from scipy.stats import pearsonr, kendalltau
from utils import get_targets, compute_graam
from measures.compute import *
from measures.pacbayes import sharpness_sigma

# removes batch normalization layers from the model
# module.children() returns the children of a module in the forward pass order. Recurssive constru  ction is allowed.
def reparam(model, prev_layer=None):
    for child in model.children():
        module_name = child._get_name()
        prev_layer = reparam(child, prev_layer)
        if module_name in ['Linear', 'Conv1d', 'Conv2d', 'Conv3d']:
            prev_layer = child
        elif module_name in ['BatchNorm2d', 'BatchNorm1d']:
            with torch.no_grad():
                scale = child.weight / ((child.running_var + child.eps).sqrt())
                prev_layer.bias.copy_( child.bias  + ( scale * (prev_layer.bias - child.running_mean) ) )
                perm = list(reversed(range(prev_layer.weight.dim())))
                prev_layer.weight.copy_((prev_layer.weight.permute(perm) * scale ).permute(perm))
                child.bias.fill_(0)
                child.weight.fill_(1)
                child.running_mean.fill_(0)
                child.running_var.fill_(1)
    return prev_layer


# This function calculates a measure on the given model
# measure_func is a function that returns a value for a given linear or convolutional layer
# calc_measure calculates the values on individual layers and then calculate the final value based on the given operator
def calc_measure(model, init_model, measure_func, operator, kwargs={}, p=1):
    measure_val = 0
    if operator == 'product':
        measure_val = math.exp(calc_measure(model, init_model, measure_func, 'log_product', kwargs, p))
    elif operator == 'norm':
        measure_val = (calc_measure(model, init_model, measure_func, 'sum', kwargs, p=p)) ** (1 / p)
    else:
        measure_val = 0
        for child, init_child in zip(model.children(), init_model.children()):
            module_name = child._get_name()
            if module_name in ['Linear', 'Conv1d', 'Conv2d', 'Conv3d']:
                if operator == 'log_product':
                    measure_val += math.log(measure_func(child, init_child, **kwargs))
                elif operator == 'sum':
                    measure_val += (measure_func(child, init_child, **kwargs)) ** p
                elif operator == 'max':
                    measure_val = max(measure_val, measure_func(child, init_child, **kwargs))
            else:
                measure_val += calc_measure(child, init_child, measure_func, operator, kwargs, p=p)
    return measure_val

# calculates l_pq norm of the parameter matrix of a layer:
# 1) l_p norm of incomming weights to each hidden unit and l_q norm on the hidden units
# 2) convolutional tensors are reshaped in a way that all dimensions except the output are together
def norm(module, init_module, p=2, q=2):
    return module.weight.view(module.weight.size(0), -1).norm(p=p, dim=1).norm(q).item()

# calculates l_p norm of eigen values of a layer
# convolutional tensors are reshaped in a way that all dimensions except the output are together
def op_norm(module, init_module, p=float('Inf')):
    _, S, _ = module.weight.view(module.weight.size(0), -1).svd()
    return S.norm(p).item()

# calculates l_pq distance of the parameter matrix of a layer from the random initialization:
# 1) l_p norm of incomming weights to each hidden unit and l_q norm on the hidden units
# 2) convolutional tensors are reshaped in a way that all dimensions except the output are together
def dist(module, init_module, p=2, q=2):
    return (module.weight - init_module.weight).view(module.weight.size(0), -1).norm(p=p, dim=1).norm(q).item()

def get_vec_params(weights: List[Tensor]) -> Tensor:
    return torch.cat([p.view(-1) for p in weights], dim=0)

# calculates l_pq distance of the parameter matrix of a layer from the random initialization with an extra factor that
# depends on the number of hidden units
def h_dist(module, init_module, p=2, q=2):
    return (n_hidden(module, init_module) ** (1 - 1 / q )) * dist(module, init_module, p=p, q=q)

# ratio of the h_dist to the operator norm
def h_dist_op_norm(module, init_module, p=2, q=2, p_op=float('Inf')):
    return h_dist(module, init_module, p=p, q=q) / op_norm(module, init_module, p=p_op)

# number of hidden units
def n_hidden(module, init_module):
    return module.weight.size(0)

# depth --> always 1 for any linear or convolutional layer
def depth(module, init_module):
    return 1

# number of parameters
def n_param(module, init_module):
    bparam = 0 if module.bias is None else module.bias.size(0)
    return bparam + module.weight.size(0) * module.weight.view(module.weight.size(0),-1).size(1)

# This function calculates path-norm introduced in Neyshabur et al. 2015
def lp_path_norm(model, device, p=2, input_size=[3, 32, 32]):
    tmp_model = copy.deepcopy(model)
    tmp_model.eval()
    for param in tmp_model.parameters():
        if param.requires_grad:
            param.abs_().pow_(p)
    data_ones = torch.ones(input_size).to(device)
    return (tmp_model(data_ones).sum() ** (1 / p )).item()


def get_weights_only(model) -> List[Tensor]:
    return [p for p in model.parameters()]

def get_reshaped_weights(weights: List[Tensor]) -> List[Tensor]:
    # If the weight is a tensor (e.g. a 4D Conv2d weight), it will be reshaped to a 2D matrix
    return [p.view(p.shape[0],-1) for p in weights]


def delta(predicted, actual):
    return 1 if predicted == actual else 0

def N(mean, var, device):
    return torch.normal(mean, torch.sqrt(torch.tensor(var).to(device))).to(device)

def set_parameters(model, parameters):
    for param, new_param in zip(model.parameters(), parameters):
        param.data = new_param.data.clone()


def pacbayes_bound(reference_vec: Tensor, m, sigma) -> Tensor:
    return ((reference_vec.norm(p=2) ** 2) / (4 * sigma ** 2) + math.log(m / sigma) + 10).item()



def calculate_measurements(model, device, tr_acc, matrix_loader, training_loader, n_images_matrix, criterion, n_train, args, matrix_0 = None, std_0 = None, sigma = None):
    
    y_matrix = get_targets(device, matrix_loader, n_images_matrix)

    d = 0.1
    if sigma is None:
        sigma = sharpness_sigma(model, training_loader, tr_acc, d)

    alpha = sigma * math.sqrt(2*math.log((2*nparam) / d))
    print(sigma)
    print(alpha)

    nparam = calc_measure(model, model, n_param, 'sum', {})

    log_prod_sum = math.log1p(compute_log_prod_sum(model, device))
    log_prod_spec = math.log1p(compute_log_prod_spec(model, device))
    fro_over_spec = math.log1p(compute_fro_over_spec(model, device))
    log_prod_fro = math.log1p(compute_log_prod_fro(model, device))
    pac_bayes = math.log1p(compute_pac_bayes(model, sigma, n_train, d, device))
    pac_bayes_sharp = math.log1p(compute_pac_bayes_sharp(model, alpha, d, n_train, nparam, device))

    # NTK based measures
    matrix = compute_graam(model, device, matrix_loader, n_images_matrix, criterion, approximation=args.approximation_k, testing=args.testing_k)
    if matrix_0 is None:
        matrix_0 = copy.deepcopy(matrix)
        std_0 = matrix_0.view(-1).std().item()
    k_mean = math.log1p(mean_matrix(matrix).item())
    k_fro = math.log1p(norm_matrix(matrix).item())
    k_corr = math.log1p(matrix_correlation(matrix, matrix_0, std_0=std_0).item())
    k_diff = math.log1p(relative_matrix_diff(matrix, matrix_0).item())
    k_lga = math.log1p(matrix_lga(matrix, y_matrix).item())

    return {
        'log_prod_sum': log_prod_sum,
        'log_prod_spec': log_prod_spec,
        'fro_over_spec': fro_over_spec,
        'log_prod_fro': log_prod_fro,
        'pac_bayes': pac_bayes,
        'pac_bayes_sharp': pac_bayes_sharp,
        'k_mean': k_mean,
        'k_fro': k_fro,
        'k_corr': k_corr,
        'k_diff': k_diff,
        'k_lga': k_lga
    }, matrix_0, std_0, sigma


def calculate(init_model, device, n, margin, learning_rate, init_mat, targets, nchannels, nclasses, img_dim):


    init_matrix = copy.deepcopy(init_mat)
    reparam(init_model)

    theta_0 = get_weights_only(init_model)
    print(type(theta_0))
    d = 0.01
    sigma_max = 2.0
    sigma_min = 1.0
    M1 = 10
    M2 = 10
    M3 = 10
    eps_d = 0.01
    eps_sigma = 0.01

    # sigma = sigma_pac_bayes(model, theta_0, l, d, sigma_max, sigma_min, M1, M2, M3, eps_d, eps_sigma, D, device, nclasses)
    sigma = 1
    print(sigma)
    seed = 42

    # depth
    d = calc_measure(init_model, init_model, depth, 'sum', {})

    # number of parameters (not including batch norm)
    nparam = calc_measure(init_model, init_model, n_param, 'sum', {})


    measure, bound = {}, {}
    with torch.no_grad():

        measure['log prod sum']
        measure['L_{1,inf} norm'] = calc_measure(init_model, init_model, norm, 'product', {'p':1, 'q':float('Inf')}) / margin
        measure['Frobenious norm'] = calc_measure(init_model, init_model, norm, 'product', {'p':2, 'q':2}) / margin
        measure['L_{3,1.5} norm'] = calc_measure(init_model, init_model, norm, 'product', {'p':3, 'q':1.5}) / margin
        measure['Spectral norm'] = calc_measure(init_model, init_model, op_norm, 'product', {'p':float('Inf')}) / margin
        measure['L_1.5 operator norm'] = calc_measure(init_model, init_model, op_norm, 'product', {'p':1.5}) / margin
        measure['Trace norm'] = calc_measure(init_model, init_model, op_norm, 'product', {'p':1}) / margin
        measure['L1_path norm'] = lp_path_norm(init_model, device, p=1, input_size=[1, nchannels, img_dim, img_dim]) / margin
        measure['L1.5_path norm'] = lp_path_norm(init_model, device, p=1.5, input_size=[1, nchannels, img_dim, img_dim]) / margin
        measure['L2_path norm'] = lp_path_norm(init_model, device, p=2, input_size=[1, nchannels, img_dim, img_dim]) / margin

        measure['mu_1 norm'] = 1 / margin
        failure_proba = 0.01
        alpha = sigma * math.sqrt(2*math.log((2*nparam) / d))
        weights = get_weights_only(init_model)
        dist_init_weights = [p-q for p,q in zip(weights, get_weights_only(init_model))]
        w_vec = get_vec_params(weights)
        dist_w_vec = get_vec_params(dist_init_weights)
        d = len(weights)

        # sharpness measures and bounds
        
        measure['Flatness Init'] = pacbayes_bound(dist_w_vec, n, sigma)
        measure['Flatness Orig'] = pacbayes_bound(w_vec, n, sigma)
        measure['Sharpness Init'] = ((calc_measure(init_model, init_model, norm, 'sum', p = 2) * math.log(2*nparam)) / (4 * alpha**2)) + math.log(n / sigma) + 10
        measure['Sharpness Orig'] = ((lp_path_norm(init_model, device, p = 3, input_size=[1, nchannels, img_dim, img_dim]) * math.log(2*nparam)) / (2 * alpha**2)) + math.log(n / d) + 10




        bound['L1_max Bound (Bartlett and Mendelson 2002)'] = alpha * measure['L_{1,inf} norm'] / math.sqrt(n)
        bound['Frobenious Bound (Neyshabur et al. 2015)'] = alpha * measure['Frobenious norm'] / math.sqrt(n)
        bound['L_{3,1.5} Bound (Neyshabur et al. 2015)'] = alpha * measure['L_{3,1.5} norm'] / ( n ** (1/3))

        beta = math.log(n) * math.log(nparam)
        ratio = calc_measure(init_model, init_model, h_dist_op_norm,'norm', {'p':2, 'q':1, 'p_op':float('Inf')}, p=2/3)
        bound['Spec_L_{2,1} Bound (Bartlett et al. 2017)'] = beta * measure['Spectral norm'] * ratio / math.sqrt(n)

        ratio = calc_measure(init_model, init_model, h_dist_op_norm,'norm', {'p':2, 'q':2, 'p_op':float('Inf')}, p=2)
        bound['Spec_Fro Bound (Neyshabur et al. 2018)'] =  d * measure['Spectral norm'] * ratio / math.sqrt(n)

        # Measures on the Gram matrix

        measure['Gram Frobenious norm'] = norm_matrix(init_matrix, p = 'fro').item()
        measure['Gram Frobenious norm over margin'] = measure['Frobenious norm'] / margin
        measure['Gram Mean'] = mean_matrix(init_matrix).item()
        measure['Gram Mean over margin'] = measure['Gram Mean'] / margin
        measure['Gram Diagonal Sum'] = torch.sum(diagonal_matrix(init_matrix)).item()
        measure['Gram Diagonal Sum over margin'] = measure['Gram Diagonal Sum'] / margin
        measure['Gram Rank'] = rank_matrix(init_matrix).item()
        measure['Gram LGA'] = matrix_lga(init_matrix, targets).item()
        measure['Gram LGA over margin'] = measure['Gram LGA'] / margin

    return measure, bound

def get_measures(measure_dicts, measures):
    return [[measure_dict[measure] for measure_dict in measure_dicts] for measure in measures]

def gradient_x(gradient, x):
    # computes the 
    return torch.dot(gradient.view(-1), x.view(-1))

def get_bounds(bound_dicts, bounds):
    return [[bound_dict[bound] for bound_dict in bound_dicts] for bound in bounds]

def sign(x):
    return 1 if x >= 0 else -1

def kendall_rank_corr_accurate(empirical_gens, measures):
    ''' Based on Jiang et al. 2019
        We build the Tau set of the measured generalization gaps, and the theoretical measure and bound of the models. We use a single
        The set is ranked by the generalization gap (the lower is ranked first).
        Accurate with the definition given by Jiang et al.

    '''
    Tau = list(zip(empirical_gens, measures))
    Tau = sorted(Tau, key=lambda x: x[0])

    # Calculate the Kendall rank correlation coefficient (sign-error)

    n = len(Tau)
    tau = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            tau += sign(Tau[i][1] - Tau[j][1]) * sign(Tau[i][0] - Tau[j][0])
    
    if n == 1:
        return tau / n
    
    tau = tau / (n * (n - 1))
    return tau

def kendall_rank_corr_adapted(empirical_gens, measures):
    '''Based on Jiang et al. 2019
        We build the Tau set of the measured generalization gaps, and the theoretical measure and bound of the models. 
        The set is ranked by the generalization gap (the lower is ranked first).
        We then compute the Kendall's rank correlation coefficient
    '''

    Tau = list(zip(empirical_gens, measures))
    Tau = sorted(Tau, key=lambda x: x[0])

    n = len(Tau)
    tau = 0
    for i in range(n-1):
        for j in range(i+1, n):
            tau += sign(Tau[i][1] - Tau[j][1]) * sign(Tau[i][0] - Tau[j][0])
    
    if n == 1:
        return tau / n
    
    tau = tau / (n * (n - 1))
    return tau


def compute_kendall(measure_dicts, bound_dicts, empirical_gens, measures, bounds, kendall_f = kendall_rank_corr_accurate):

    measure_vals = get_measures(measure_dicts, measures)
    bound_vals = get_bounds(bound_dicts, bounds)

    measure_kendall = []
    bound_kendall = []
    for measures in measure_vals:
        measure_corr = kendall_f(empirical_gens, measures)
        measure_kendall.append(measure_corr)
    
    for bounds in bound_vals:
        bound_corr = kendall_f(empirical_gens, bounds)
        bound_kendall.append(bound_corr)

    return measure_kendall, bound_kendall

