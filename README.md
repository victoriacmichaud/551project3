# Group 65 COMP 551 Project 3

18 Mar 2018

[Data was available here](https://www.kaggle.com/c/comp-551-w2019-project-3-modified-mnist/data)

## Overview of files

### Python

- `resnet_mnist.py`: train resnet models
- `resnet_mnist_submission.py`: load trained resnet weights and write a
  submission csv for kaggle
- `MnistResNet.py`: contains the modified ResNet classes used
- `util.py`: dependency, contains Dataset and DataLoader classes as well as
  helper functions
- `Preprocessing.py:` write our 3 additional preprocessed datasets for use with
  `SVM.py` (must be run **before** `SVM.py`)
- `SVM.py`: train linear SVM classifiers on the raw and preprocessed datasets

### Others

- `models`: contains the PyTorch model weights (`.pt`) of our top two kaggle
  submissions
- `data/input`: location where the scripts expect the raw data to be,
  preprocessed data will be written here
- `data/output`: kaggle model submissions and svm outputs will be written here 
- `enviornment.yml`: conda environment used during the kaggle submissions

## Kaggle Submission Replication

Our top two kaggle models were trained using `resnet_mnist.py` and submission
csvs were created using `resnet_mnist_submission.py`. Both these scripts take
arguments which can be see by running, eg, `./resnet_mnist.py --help`. These are
the commands used to generate our kaggle submissions (the default random seed of
1 was used):

```bash
# Submission 1
./resnet_mnist.py --epochs 125 --save-model resnet152_mnist-e125.pt --depth 152
./resnet_mnist_submission.py resnet152_mnist-e125.pt --depth 152

# Submission 2
./resnet_mnist.py --epochs 100 --save-model wide_resnet_mnist-e100.pt --depth 50
./resnet_mnist_submission.py wide_resnet_mnist-e100.pt --depth 50
```
