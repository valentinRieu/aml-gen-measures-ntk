# Analysis of Machine Learning model using Generalization Measures -- a Neural Tangent Kernel view


---

## Installation

This framework has been tested on Ubuntu 20.04, with Python 3.9, Pytorch 1.11.0, cudatoolkit 10.2, torchvision 0.12.0, scipy 1.13.1,

I recommend to setup a new conda environment, with the given packages:

1. If you plan to use the GPU:
```bash
conda create -n generalization-measures python==3.9 pytorch==1.11.0 torchvision==0.12.0 matplotlib cudatoolkit=10.2 -c pytorch
```

2. If you plan to use only the CPU:
```bash
conda create -n generalization-measures python==3.9 pytorch==1.11.0 torchvision==0.12.0 matplotlib cpuonly -c pytorch
```

Once the conda environment created, activate it using `conda activate generalization-measures`, and install `scipy`:

```bash
pip install scipy
```

---

## Training

The main file is `train.py`, that trains several models and perform measurements at each epoch.

### Arguments
To control the experiment, you provide arguments to the file. Here is the list:

```bash
--name		Name of the experiment. Default is the list of varying hyperparameters

--seed		Seed of the experiment. Default is random

--no-cuda		Deactivates GPU training. Do not specify for GPU training

--dataset		Dataset used for training. Default is MNIST.

--model		Machine Learning model used for training. Default is fc

--datadir		Directory where the datasets are stored. Default is ./datasets

--n-nn		Number of neural networks in the experiment.

--epochs		Maximum number of epochs of training.

--stopcond	Early Stopper's tolerance. Default is 0.01

--batchsize	Number of samples in each batch. Default is 64

--batches		Number of batches. Default is the whole dataset.

--save-every	Compute measurements after n epochs of training. Default is 1

--init-method:	Weight Initializer method. default is xavier_uniform_. See [torch.nn.init] for the complete list

--weight-decay Weight decay penaly for the optimizer. Default is 0.001
```


### Hyperparameters

The following arguments are the main hyperparameters, that will define the experiment. By setting the argument to a negative value, the hyperparameter will cycle between a selection of values, specified in the csv files from the `csv` folder, with one unique value for each of the models of the experiment. Fixing the value will fix it for all the models.
```bash
--learning-rate	Learning rate of the training procedure. Default is -1
```

### FCNN-only hyperparameters

The following arguments have a meaning only when training FC neural networks, i.e. when `--model` is `fc`. 
```bash
--depth	Specifies the number of hidden layers in the NN. Default is -1
--width	Specifies the number of neurons in each hidden layer. Default is -1
--dropout	Specifies the dropout rate applied to the hidden layers. Default is -1
```

## Example

If you need to recall the arguments, do
```bash
python3 train.py --help
```
To print usage indications.

To train a simple model on the CPU and the MNIST dataset, with fixed hyperparameters, we can use the following command:
```bash
python3 train.py --dataset MNIST --model fc --n-nn 1  --epochs 100 --width 256 --depth 1 --dropout  0.0 --learning-rate 0.01 --no-cuda --init-method xavier_normal_ --seed 42
```

To train 4 models, with a varying depth, on the GPU:

```bash
python3 train.py --dataset MNIST --model fc --n-nn 1  --epochs 100 --width 256 --depth -1 --dropout  0.0 --learning-rate 0.01  --init-method xavier_normal_ --seed 42
```


















































