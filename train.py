import copy
import math
import random
import numpy as np
import torch
from models.fc import Network
from torch import nn, optim
from torch.utils.data import DataLoader, SequentialSampler
import importlib
from measures.measures import calculate_measurements
import matplotlib.pyplot as plt

from arguments import get_arguments
from utils import load_parameter, load_data
from utils import validate, train, compute_graam
from utils import get_targets

import os
from EarlyStop import EarlyStop

def comma_num(n,f=''):
    return ('{'+f+'}').format(n).replace('.',',')



def plot_error(experiment, epochs, data, y_name, x_name = 'epoch', bounds = None):
    # Find the length of the shortest list in data
    min_length = min(len(lst) for lst in data)

    # Limit the range of epochs and the length of each list in data
    epochs = epochs[:min_length]
    data = [lst[:min_length] for lst in data]

    fig, ax = plt.subplots()

    for i, y in enumerate(data):
        ax.plot(epochs, y, color=f"C{i}", label=f"NN_{i+1}")

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)

    if bounds:
        ax.set_ylim(bounds)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(f'experiments/plots_final/{experiment}_{y_name}.png')


def plot_combined(experiment, epochs, data1, data2, y2_name, x_name = 'epoch', y1_name = 'Error', bounds = None):
    # Find the length of the shortest list in data1

    min_length = min(len(lst) for lst in data1)

    # Limit the range of epochs and the length of each list in data1 and data2
    epochs = epochs[:min_length]
    data1 = [lst[:min_length] for lst in data1]
    data2 = [lst[:min_length] for lst in data2]

    fig, ax1 = plt.subplots()

    for i, y in enumerate(data1):
        ax1.plot(epochs, y, color=f"C{i}", label=f"Data1_{i}")

    ax1.set_xlabel(x_name)
    ax1.set_ylabel(y1_name)

    # Second plot
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    for i, y in enumerate(data2):
        ax2.plot(epochs, y, color=f"C{i}", linestyle='dashed', label=f"Data2_{i}")

    ax2.set_ylabel(y2_name)

    if bounds:
        ax1.set_ylim(bounds)
        ax2.set_ylim(bounds)

    fig.tight_layout()
    plt.savefig(f'experiments/plots_final/{experiment}_{y1_name}_{y2_name}.png')


def main():

    args = get_arguments()

    seed = args.seed
    if args.seed <= -1:
        seed = np.random.randint(0, 1048576)


    def worker_init_fn(worker_id):                                                                                                                                
        wseed = seed             
                                                                                                                                    
        torch.manual_seed(wseed)                                                                                                                                   
        torch.cuda.manual_seed(wseed)                                                                                                                              
        torch.cuda.manual_seed_all(wseed)                                                                                          
        np.random.seed(wseed)                                                                                                             
        random.seed(wseed)                                                                                                                                
    
    
    print(f'Using seed {seed}')
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    fixed = ''
    experiment = ''
    if args.name != '':
        fixed += f'fixed_{args.name}'
        experiment = f'{args.name}'
    else:
        if args.depth > -1: fixed += f'_d{args.depth}'
        else: experiment += 'depth_'
        if args.width > -1: fixed += f'_w{args.width}'
        else: experiment += 'width_'
        if args.dropout > -1: fixed += f'_do{comma_num(args.dropout)}'
        else: experiment += 'dropout_'
        if args.learning_rate > -1: fixed += f'_lr{comma_num(args.learning_rate)}'
        else: experiment += 'learning_rate_'

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    nchannels, nclasses,  = 3, 10
    img_dim = 32
    if args.dataset == 'MNIST': nchannels = 1
    if args.dataset == 'CIFAR100': nclasses = 100
    input_size = nchannels * img_dim * img_dim


    # Load the hyperparameters for each models: either the provided argument, or the csv file
    nhidden = load_parameter('depth', args.depth, args.n_nn, int)
    width = load_parameter('width', args.width, args.n_nn, int)
    dropout = load_parameter('dropout', args.dropout, args.n_nn)
    learningrate = load_parameter('learning_rate', args.learning_rate, args.n_nn)
    weight_decay = load_parameter('weight_decay', args.weight_decay, args.n_nn)
    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    print('nhidden:', nhidden)
    print('width:', width)
    print('dropout:', dropout)
    print('learningrate:', learningrate)
    print('weight_decay:', weight_decay)



    train_dataset = load_data('train', args.dataset, args.datadir, nchannels)
    val_dataset = load_data('val', args.dataset, args.datadir, nchannels)

    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, worker_init_fn=worker_init_fn, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False, **kwargs)


    matrix_sampler = SequentialSampler(train_dataset)
    matrix_loader = DataLoader(train_dataset, batch_size=1, sampler=matrix_sampler, shuffle=False)


    n_train = len(train_loader)
    n_images_matrix = math.ceil(len(train_loader))
    print(n_images_matrix)

    if not os.path.exists('experiments/targets'):
        os.makedirs('experiments/targets')

    if os.path.exists(f'experiments/targets/{args.dataset}_{seed}.pt'):
        y_matrix = torch.load(f'experiments/targets/{args.dataset}_{seed}.pt', map_location = device)
    else:
        y_matrix = get_targets(device, matrix_loader, n_images_matrix).to(device)
        torch.save(y_matrix, f'experiments/targets/{args.dataset}_{seed}.pt')

    
    n_val = len(val_loader)
    print(n_train * args.batchsize, n_val * args.batchsize)
    train_batches = n_train if args.batches <= -1 or args.batches > n_train else args.batches
    val_batches = len(val_loader) if args.batches <= -1 or args.batches > len(val_loader) else args.batches


    tr_errss, val_errss = [], []
    # Classic measures
    log_prod_sums, log_prod_specs, fro_over_specs, log_prod_fros = [], [], [], []
    pac_bayess, pac_bayes_sharps = [], []

    # NTK based measures
    k_means, k_fros = [], []
    k_corrs, k_diffs, k_lgas = [], [], []
    savery = max(1, args.save_every)
        
    epochs = range(savery, args.epochs + 1)
    print("Epochs:", len(epochs))
    if not os.path.exists('experiments/plots_final/'):
        os.makedirs('experiments/plots_final/')
    
    print('Experiment:', experiment)

    for i in range(args.n_nn):
        current_nn = f'NN{i+1}_d{nhidden[i]}_w{width[i]}_do{comma_num(dropout[i])}_lr{comma_num(learningrate[i])}'

        print('-'*10)
        print('\n')
        print(f'Training the NN {i+1}\n with {nhidden[i]} hidden layers, width {width[i]}, dropout {dropout[i]}, learning rate {learningrate[i]}')
        print(args.init_method)
        model : Network  = getattr(importlib.import_module('models.{}'.format(args.model)), 'Network')(nchannels, img_dim, nclasses, width[i], nhidden[i], dropout[i], args.init_method)
        model = model.to(device)

        delta = 0.01
        sigma = 0.1
        nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(nparam)
        alpha = sigma * math.sqrt(2*math.log((2*nparam) / delta))
        print(alpha)
        optimizer = optim.Adam(model.parameters(), learningrate[i], weight_decay= weight_decay[i])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.1, 5)
        stopper = EarlyStop(10, 0)


        tr_errs, val_errs = [], []
        # Classic measures
        log_prod_sum, log_prod_spec, fro_over_spec, log_prod_fro = [], [], [], []
        pac_bayes, pac_bayes_sharp = [], []

        # NTK based measures
        k_mean, k_fro = [], []
        k_corr, k_diff, k_lga = [], [], []
        count_save = 0
        passed_epochs = 0
        saved_init = False

        matrix_0 = None
        std_0 = None
        sigma = None
        if args.save_every <= 0:
            print('Computing the initial matrix before training')
            matrix_0 = compute_graam(model, device, matrix_loader, n_images_matrix, criterion, approximation=args.approximation_k, testing=args.testing_k)
            std_0 = matrix_0.std().item()


        for epoch in range(1, args.epochs):

            tr_acc, tr_err, tr_loss = train(args, model, device, train_loader, criterion, optimizer, epoch, train_batches)

            _, val_err, val_loss, val_margin = validate(args, model, device, val_loader, criterion, val_batches)

            print(f'NN {i+1}\t weight_decay = {weight_decay[i]}, Epoch: {epoch}/{args.epochs}\t Training loss: {tr_loss:.3f}\t',
                    f'Training error: {tr_err:.3f}\t Validation error: {val_err:.3f}\t Validation loss: {val_loss:.3f}\t Validation margin: {val_margin:.3f}')
            scheduler.step(val_loss)


            save_gradient = (epoch >= args.save_every)
            if save_gradient:
                print('Computing measurements at epoch', epoch)
                dict_meas, new_matrix_0, new_std_0, new_sigma = calculate_measurements(model, device, nchannels, img_dim, val_margin, tr_acc, matrix_loader, train_loader, n_images_matrix, criterion, n_train, args, matrix_0=matrix_0, std_0=std_0, sigma=sigma)
                sigma = new_sigma
                std_0 = new_std_0 
                matrix_0 = copy.deepcopy(new_matrix_0)
                new_matrix_0 = None

                # Computing the different measures
                # Classic measures
                tr_errs.append(tr_err)
                val_errs.append(val_err)

                log_prod_sum.append(dict_meas['log_prod_sum'])
                log_prod_spec.append(dict_meas['log_prod_spec'])
                fro_over_spec.append(dict_meas['fro_over_spec'])
                log_prod_fro.append(dict_meas['log_prod_fro'])
                pac_bayes.append(dict_meas['pac_bayes'])
                pac_bayes_sharp.append(dict_meas["pac_bayes_sharp"])

                k_mean.append(dict_meas['k_mean'])
                k_fro.append(dict_meas['k_fro'])
                k_corr.append(dict_meas['k_corr'])
                k_diff.append(dict_meas['k_diff'])
                k_lga.append(dict_meas['k_lga'])
            
                matrix = None
                passed_epochs += 1

            if stopper.step(val_err):
                break
        
        # fill tr_err until the last epoch
        print(len(epochs) - len(tr_errs))
        tr_errs += [tr_errs[-1]]
        print(len(epochs) - len(val_errs))
        val_errs += [val_errs[-1]]
        print(len(epochs) - len(log_prod_sum))
        log_prod_sum += [log_prod_sum[-1]]
        print(len(epochs) - len(log_prod_spec))
        log_prod_spec += [log_prod_spec[-1]]
        print(len(epochs) - len(fro_over_spec))
        fro_over_spec += [fro_over_spec[-1]]
        log_prod_fro += [log_prod_fro[-1]]
        pac_bayes += [pac_bayes[-1]]
        pac_bayes_sharp += [pac_bayes_sharp[-1]]
        print(len(epochs) - len(k_mean))
        k_mean += [k_mean[-1]]
        k_fro += [k_fro[-1]]
        k_corr += [k_corr[-1]]
        k_diff += [k_diff[-1]] 
        k_lga += [k_lga[-1]]

        tr_errss.append(tr_errs)
        val_errss.append(val_errs)
        log_prod_sums.append(log_prod_sum)
        log_prod_specs.append(log_prod_spec)
        fro_over_specs.append(fro_over_spec)
        log_prod_fros.append(log_prod_fro)
        pac_bayess.append(pac_bayes)
        pac_bayes_sharps.append(pac_bayes_sharp)

        k_means.append(k_mean)
        k_fros.append(k_fro)
        k_corrs.append(k_corr)
        k_diffs.append(k_diff)
        k_lgas.append(k_lga)
    
    # Plotting the combined validation error + measures


    plot_error(experiment, epochs, val_errss, 'Validation_Error')
    plot_error(experiment, epochs, tr_errss, 'Training_Error')
    plot_error(experiment, epochs, k_lgas, 'k.lga')
    plot_error(experiment, epochs, k_diffs, 'k.diff')
    plot_error(experiment, epochs, k_corrs, 'k.corr')
    plot_error(experiment, epochs, k_fros, 'k.fro')
    plot_combined(experiment, epochs, tr_errss, val_errss, 'Validation_error', y1_name = 'Training_Error', bounds= (0.20, 0.58))

    plot_combined(experiment, epochs, val_errss, log_prod_sums, 'log.prod.sum', y1_name = 'Validation_Error')
    plot_combined(experiment, epochs, val_errss, log_prod_specs, 'log.prod.spec', y1_name = 'Validation_Error')
    plot_combined(experiment, epochs, val_errss, fro_over_specs, 'fro.over.spec', y1_name = 'Validation_Error')
    plot_combined(experiment, epochs, val_errss, log_prod_fros, 'log.prod.fro', y1_name = 'Validation_Error')
    plot_combined(experiment, epochs, val_errss, pac_bayess, 'pac.bayes', y1_name = 'Validation_Error')
    plot_combined(experiment, epochs, val_errss, pac_bayes_sharps, 'pac.bayes.sharp', y1_name = 'Validation_Error')
    plot_combined(experiment, epochs, val_errss, k_means, 'k.mean', y1_name = 'Validation_Error')
    plot_combined(experiment, epochs, val_errss, k_fros, 'k.fro', y1_name = 'Validation_Error')
    plot_combined(experiment, epochs, val_errss, k_corrs, 'k.corr', y1_name = 'Validation_Error')
    plot_combined(experiment, epochs, val_errss, k_diffs, 'k.diff', y1_name = 'Validation_Error')
    plot_combined(experiment, epochs, val_errss, k_lgas, 'k.lga', y1_name = 'Validation_Error')



if __name__ == '__main__':
    main()