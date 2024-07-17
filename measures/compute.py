import torch
from torch import Tensor
import math

def norm_matrix(matrix, p = 'fro'):
    return torch.norm(matrix, p = p)

def mean_matrix(matrix: Tensor):
    return torch.div(torch.sum(matrix), (matrix.shape[0] * matrix.shape[1]))

def diagonal_matrix(matrix: Tensor):
    return torch.diagonal(matrix)

def rank_matrix(matrix: Tensor, symmetric = True):
    return torch.matrix_rank(matrix, symmetric= symmetric)

def matrix_correlation(matrix_t, matrix_0, std_t = None, std_0 = None):
    # Pearson correlation coefficient
    if std_t is None:
        std_t = torch.std(matrix_t.view(-1)).item()
    
    if std_0 is None:
        std_0 = torch.std(matrix_0.view(-1)).item()
    
    mat_t_f = matrix_t.view(-1)
    mat_0_f = matrix_0.view(-1)

    mat_t_mean = matrix_t.mean().item()
    mat_0_mean = matrix_0.mean().item()
    
    corr = torch.mean((mat_t_f - mat_t_mean) * (mat_0_f - mat_0_mean)) / (std_t * std_0)

    return corr

def relative_matrix_diff(matrix_t, matrix_0):
    return (torch.abs(matrix_t - matrix_0) / torch.abs(matrix_0)).sum()

def trace(matrix):
    return torch.trace(matrix)

def det(matrix):
    return torch.det(matrix)

def sum_matrix(matrix):
    return torch.sum(matrix)

def mean_matrix(matrix):
    return torch.mean(matrix)

def matrix_lga(matrix, targets):
    A = torch.where(targets == targets.T, 1, -1).to(torch.float32)
    A_centered = A - torch.mean(A, dim = 0)
    m_centered = matrix - torch.mean(matrix)
    norm_m = torch.norm(m_centered, p = 2)
    norm_a = torch.norm(A_centered, p = 2)
    return torch.frac(torch.dot(m_centered.view(-1), A_centered.view(-1)) / (norm_m * norm_a))

def compute_log_prod_sum(model, device):
    spectral_norm_product = torch.tensor(1.0).to(device)
    frobenius_norm_sum = torch.tensor(0.0).to(device)

    for param in model.parameters():
        if param.dim() >= 2:  # Only consider parameters that have at least 2 dimensions
            spectral_norm = torch.linalg.norm(param, ord=2).to(device)
            frobenius_norm = torch.linalg.norm(param, ord='fro').to(device)

            spectral_norm_product *= spectral_norm**2
            frobenius_norm_sum += (frobenius_norm**2) / (spectral_norm**2)

    measure = spectral_norm_product * frobenius_norm_sum

    return measure.item()

def compute_log_prod_spec(model, device):
    spectral_norm_product = torch.tensor(1.0).to(device)

    for param in model.parameters():
        if param.dim() >= 2:  # Only consider parameters that have at least 2 dimensions
            spectral_norm = torch.linalg.norm(param, ord=2).to(device)
            spectral_norm_product *= spectral_norm**2

    return spectral_norm_product.item()

def compute_fro_over_spec(model, device):
    fro_over_spec = torch.tensor(0.0).to(device)

    for param in model.parameters():
        if param.dim() >= 2:  # Only consider parameters that have at least 2 dimensions
            spectral_norm = torch.linalg.norm(param, ord=2).to(device)
            frobenius_norm = torch.linalg.norm(param, ord='fro').to(device)

            frobenius_norm_sq = frobenius_norm**2
            spectral_norm_sq = spectral_norm**2
            fro_over_spec += frobenius_norm_sq / spectral_norm_sq

    return fro_over_spec.item()


def compute_log_prod_fro(model, device):
    frobenius_norm_product = torch.tensor(1.0).to(device)

    for param in model.parameters():
        if param.dim() >= 2:  # Only consider parameters that have at least 2 dimensions
            frobenius_norm = torch.linalg.norm(param, ord='fro').to(device)
            frobenius_norm_product *= frobenius_norm**2

    return frobenius_norm_product.item()


def compute_spec(model, device):
    spectral_norm_product = torch.tensor(0.0).to(device)

    for param in model.parameters():
        if param.dim() >= 2:  # Only consider parameters that have at least 2 dimensions
            spectral_norm = torch.linalg.norm(param, ord=2).to(device)
            spectral_norm_product += spectral_norm**2

    return spectral_norm_product.item()



def compute_pac_bayes(model, sigma, n, delta, device):
    log_n_s = math.log(n / delta)
    spec_over_sigma = compute_spec(model, device) / (4 * (sigma**2))

    return (spec_over_sigma + log_n_s)



def compute_pac_bayes_sharp(model, alpha, delta, n, nparam, device):
    log_n_delta = math.log(n / delta)

    spec_over_alpha = (compute_spec(model, device) * math.log(2 * nparam)) / (4 * (alpha**2))

    return spec_over_alpha + log_n_delta