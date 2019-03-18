# Utility functions
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import inspect
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    """ torch Dataset generator (to feed to a DataLoader) for our MNIST style
    data
    X : np.ndarray, images of shape (N, 64, 64)
    Y : a df of the train_labels.csv, with columns [Id, Category] eg output of
        pd.read_csv('data/input/train_labels.csv')
    """
    def __init__(self, X, labels, transform = transforms.ToTensor()):
        self.X = X
        self.labels = labels
        self.ID = np.array(self.labels['Id'])
        self.Y = np.array(self.labels['Category'])
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """ When the sample corresponding to a given index is called, the
        generator executes the __getitem__ method to generate it.
        """
        x = self.X[idx]
        y = self.Y[idx]
        ID = self.ID[idx]

        if self.transform:
            x = self.transform(x)

        # but what about id and label?
        x.requires_grad_(True)
        return x, y
    
    def tensor_x(self):
        return self.transform(self.X)
    
    def tensor_y(self):
        return self.transform(self.Y)

class SubmissionDataset(Dataset):
    """ torch Dataset generator (to feed to a DataLoader) for our MNIST style
    data
    X : np.ndarray, images of shape (N, 64, 64)
    Y : a df of the train_labels.csv, with columns [Id, Category] eg output of
        pd.read_csv('data/input/train_labels.csv')
    """
    def __init__(self, X, transform = transforms.ToTensor()):
        self.X = X
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """ When the sample corresponding to a given index is called, the
        generator executes the __getitem__ method to generate it.
        """
        x = self.X[idx]

        if self.transform:
            x = self.transform(x)

        # but what about id and label?
        x.requires_grad_(True)
        return x
    
    def tensor_x(self):
        return self.transform(self.X)
    
def load_submission_data():
    X = pd.read_pickle('data/input/test_images.pkl')

    Transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((X.mean()/255,), (X.std()/255,))
    ])

    # X = Transformer(X)
    X_loader = DataLoader(SubmissionDataset(X, Transformer), batch_size=len(X), shuffle=False)

    return X_loader

def load_data(train_porportion=0.8, split_train=True, include_ids=False):
    """ Split the training data in training/valid sets
    include_ids: 
        true: Ytest/train are returned as dataframes with columns [Id,
        Category]
        false: Ytest/train are returned as np arrays containing just Category
    """

    train_images = pd.read_pickle('data/input/train_images.pkl')
    train_labels = pd.read_csv('data/input/train_labels.csv')

    if split_train:
        if not include_ids:
            train_labels = np.array(train_labels['Category'])

        N = len(train_images)
        last_train_idx = int(N*train_porportion)
        Y = np.squeeze(train_labels[:last_train_idx])
        testY = np.squeeze(train_labels[last_train_idx:])
        X = train_images[:last_train_idx]
        testX = train_images[last_train_idx:]
        return X, Y, testX, testY

    else:
        return train_images, train_labels

def data_loaders(train_params, test_params, transfer_learning=False):
    """ Return torch.util.data.DataLoader objects for train and valid sets

    <train/test>_params should contain at least these keys:
        'batch_size': <int>
        'shuffle': <bool>
    """
    X, Y, Xtest, Ytest = load_data(include_ids=True)

    # Use full training set to calculate metrics for normalizations
    X_full = np.vstack((X, Xtest))

    if transfer_learning:
        # Repeat the greyscale channel 3x to make an rgb im
        Transformer = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),
            transforms.Normalize((X_full.mean()/255,), (X_full.std()/255,)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])

    else:
        Transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((X_full.mean()/255,), (X_full.std()/255,))
        ])

    train_loader = DataLoader(CustomDataset(X, Y, Transformer), **train_params)
    test_loader = DataLoader(CustomDataset(Xtest, Ytest, Transformer), **test_params)

    return train_loader, test_loader

def calculate_metric(metric_fn, true_y, pred_y):
    # multi class problems need to have averaging method
    if "average" in inspect.getfullargspec(metric_fn).args:
        return metric_fn(true_y, pred_y, average="macro")
    else:
        return metric_fn(true_y, pred_y)
    
def print_scores(p, r, f1, a, batch_size):
    # just an utility printing function
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        print(f"\t{name.rjust(14, ' ')}: {sum(scores)/batch_size:.4f}")

def append_metrics(p, r, f1, a, batch_size, metrics):
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        metrics[name].append(sum(scores)/batch_size)
    return metrics

def plot_f1(metrics, kind='F1'):
    epochs = len(metrics[kind])
    y = metrics[kind]
    x_ax = range(1, epochs)
    plt.title('Validation ' + kind)
    plt.xlabel('Epochs')
    plt.ylabel(kind)
    plt.plot(x_ax, y)
    plt.xticks(np.arange(1, x_ax[-1], 1))
    plt.show()

