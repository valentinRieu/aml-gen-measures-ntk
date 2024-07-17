import argparse

def get_arguments():

    parser = argparse.ArgumentParser(description='train neural networks on a given dataset and calculates various measures on the learned networks')
    parser.add_argument('--seed', type=int, default=-1, help='pseudo-random seed (default: -1 for random seed, >=0 for fixed seed)')
    parser.add_argument('--name', type=str, default='', help='name of the experiment (default: no name)')
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset to train on. Options: MNIST | CIFAR10 | CIFAR100 | SVHN  (default: MNIST)')
    parser.add_argument('--datadir', type=str, default='./datasets', help='directory of the dataset (default: ./datasets)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--n-nn', type=int, default=1, help='number of neural networks to train (default: 1)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--batchsize', type=int, default=64, help='batch size for training (default: 64)')
    parser.add_argument('--batches', type=int, default=-1, help='number of batches per epoch (default: -1 for all batches)')
    parser.add_argument('--model', type=str, default='fc', help='neural network model to train. Options: fc | vgg (default: fc)')

    parser.add_argument('--depth', type=int, default=-1, help='number of hidden layers (default: -1 for file)')
    parser.add_argument('--width', type=int, default=-1, help='width of the neural network (default: -1 for file)')
    parser.add_argument('--dropout', type=float, default=-1, help='dropout rate (default: -1 for file)')
    parser.add_argument('--stopcond', type=float, default=0.01, help='early stopping condition (default: 0.01)')
    # parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.001, help='weight decay for optimizer (default: 0.001) -1 for file')

    parser.add_argument('--learning-rate', type=float, default=0.01, help='learning rate (default: 0.01)')
    # parser.add_argument('--no-save', action='store_true', default=False, help='disables all saving and measure computation')
    parser.add_argument('--save-every', type=int, default=1, help='save model after the n epoch (default: 1)')
    # parser.add_argument('--approximation-k', action='store_true', default=False, help='use approximation for the kernel matrix (default: false)')
    # parser.add_argument('--testing-k', action='store_true', default=False, help='tests on kernel matrix (default: false)')
    # parser.add_argument('--redo', action='store_true', default=True, help='Allow to redo an experiment. If true, deletes the previous experiment (default = True)')

    parser.add_argument('--init-method', type=str, default='xavier_uniform_', help="Weight Initializer. See torch.nn.init for options (default = xavier_uniform_)")

    return parser.parse_args()
