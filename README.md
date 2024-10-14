# Analysis of Machine Learning model using Generalization Measures -- a Neural Tangent Kernel view

---

## Installation

This framework has been tested on Ubuntu 20.04, with Python 3.9, Pytorch 1.11.0, cudatoolkit 10.2, torchvision 0.12.0, scipy 1.13.1.

I recommend to setup a new conda environment, with the given packages:


1. Ideally, create a new conda environment with Python, Pytorch, TorchVision and matplotlib.
- If using the GPU:
```bash
conda create -n generalization-measures python==3.9 pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=10.2 matplotlib -c pytorch
```

- If using only the CPU:
```bash
conda create -n generalization-measures python==3.9 pytorch==1.11.0 torchvision==0.12.0 cpuonly matplotlib -c pytorch
```
2. Activate the conda environment, and install `scipy` using `pip`
```bash
conda activate generalization-measures
pip install scipy
```

3. Clone the repository:
```bash
git clone https://github.com/valentinRieu/aml-gen-measures-ntk
```



## Training
4. Print the help of the training function:
```bash
python3 train.py --help
```
This will print the instructions

### Main arguments:

Those arguments applies to all the neural networks trained in the experiments, and define the protocol.

- `--model`: Neural network model used for the experiment. Default is fc for Fully-Connected. Limited support of cnn, for Convolutional Neural Network. 

- `--seed`: Seed used for the experiment. Default is -1 for random seed.

- `--name`: Name of the experiment for directory creations. Default is the varying hyperparameter (See next subsection).

- `--dataset`: The dataset that will be used for the experiment. Default is `MNIST`, covered options are `CIFAR10 | CIFAR100 | SVHN`.

- `--n-nn`: Number of neural networks in the experiment. Default is 1

- `--epochs`: Number of training epochs. Default is 100

- `--batchsize`: Batch size during training. Default is 64.

- `--batches`: Number of batches per epoch. Default is all batches on the datasets.

- `--datadir`: Directory where the datasets are located. Default is `./datasets`

- `--no-cuda`: Disables GPU training. Do not specify if you want to train on the GPU

- `--stopcond`: Specify the Early Stopper's tolerance. Default is 0.01

- `--weight-decay`: Specify the weight decay penalty for the optimizer. Default is 0.001

- `--save-every`: Specify at which epoch we start performing measurements. Default is 1.

- `--init-method`: Specify the Weight Initializer method. See [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html) for the list of options. Default is `xavier_uniform_`

### Hyperparameters

Those arguments specify the hyperparameters of the trained neural networks. We can either fix them to a given value, that will be identical for all models, or make them vary using values from csv files (see the folder `csv`), by specifying a negative value.

#### Common hyperparameters

- `--learning-rate`: Specify the learning rate. Default is -1 for csv

- `--dropout`: Specify the dropout rate applied to the hidden layers. Default is -1 for csv

#### FCNN-specific hyperparameters

- `--depth`: Specify the number of hidden layers. Default is -1 for csv
- `--width`: Specify the number of neurons per hidden layer. Default is -1 for csv.

## Example

A simple training of a unique model with fixed hyperparameters and done on the CPU can be written like this:

```bash
python3 train.py --model fc --name simple_training --dataset MNIST --n-nn 1 --epochs 100 --depth 1 --width 512 --dropout 0.0 --learning-rate 0.01 --save-every 1 --init-method xavier_uniform_ --seed 42 --no-cuda
```

For training 4 models with one varying hyperparameter, on the GPU:

```bash
python3 train.py --model fc --name simple_training --dataset MNIST --n-nn 4 --epochs 100 --depth 1 --width -1 --dropout 0.0 --learning-rate 0.01 --save-every 1 --init-method xavier_uniform_ --seed 42
```

## Evaluation

In the current version, plotting and evaluation of the measures are done in the same process. Next version will provide a separate file for pltotting and evaluation.

### Datasets

Right now, covered datasets are `MNIST, CIFAR10, CIFAR100, SVHN` from the [`torchvision` datasets](https://pytorch.org/vision/0.12/datasets.html). There are other built-in datasets in `torchvision`, feel free to modify the function [`utils.load_data`]. You can also add custom datasets, as long as they inherit from [`torch.utils.data.Dataset`](https://pytorch.org/docs/1.11/data.html?highlight=torch%20utils%20data%20dataset#torch.utils.data.Dataset), which can be used in a [`DataLoader`](https://pytorch.org/docs/1.11/data.html?highlight=torch%20utils%20data%20dataset#torch.utils.data.DataLoader). Loading custom datasets requires an adaptation of `utils.load_data`.

























