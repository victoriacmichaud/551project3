#!/usr/bin/env python3

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import warnings

# Try to make tensorboardX work
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

# Local imports
import util
from MnistResNet import MnistResNet152 as MnistResNet
import MnistResNet as Nets

def main():
    args = parse_args()

    # Create a Writer object for tensorboard
    if args.tensorboard:
        writer = SummaryWriter(args.tensorboard)
    else: 
        writer = SummaryWriter()

    # Option to suppress warnings from output
    if args.ignore_warnings:
        warnings.filterwarnings('ignore') 

    # Seed for repeatability
    torch.manual_seed(args.seed)


    start_ts = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # Init model and move to gpu, if possible
    model = select_resnet(args.depth)
    model.to(device)
    if args.load:
        model.load_state_dict(torch.load(args.load))
        print('Loaded model weights from', args.load)

    # Get the DataLoaders for train and validation (test) sets
    train_params = {'batch_size': args.batch_size, 'shuffle': True}
    test_params = {'batch_size': args.test_batch_size, 'shuffle': False}
    train_loader, val_loader = util.data_loaders(test_params, train_params,
                                                train_file=args.train_ims)

    # your loss function, cross entropy works well for multi-class problems
    loss_function = nn.CrossEntropyLoss() 

    # optimizer, I've used Adadelta, as it works well without any magic numbers
    optimizer = optim.Adadelta(model.parameters())

    losses = []
    metrics = {'precision': [],
               'recall':[],
               'F1': [],
               'accuracy': []}
    batches = len(train_loader)
    val_batches = len(val_loader)

    # loop for every epoch (training + evaluation)
    epochs = args.epochs
    for epoch in range(epochs):
        total_loss = 0

        # progress bar (works in Jupyter notebook too!)
        progress = tqdm(enumerate(train_loader), desc="Loss: ", total=batches)

        # ----------------- TRAINING  -------------------- 
        # set model to training
        model.train()
        
        for i, data in progress:
            X, y = data[0].to(device), data[1].to(device)
            
            # training step for single batch
            model.zero_grad()
            outputs = model(X)
            loss = loss_function(outputs, y)
            loss.backward()
            optimizer.step()

            # getting training quality data
            current_loss = loss.item()
            total_loss += current_loss

            # updating progress bar
            progress.set_description("Epoch %d/%d " %(epoch+1, epochs)
                +"Loss: {:.4f}".format(total_loss/(i+1)))
            
        # releasing unceseccary memory in GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # ----------------- VALIDATION  ----------------- 
        val_losses = 0
        precision, recall, f1, accuracy = [], [], [], []
        
        # set model to evaluating (testing)
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                X, y = data[0].to(device), data[1].to(device)

                outputs = model(X) # this get's the prediction from the network

                val_losses += loss_function(outputs, y)

                predicted_classes = torch.max(outputs, 1)[1] # get class from network's prediction
                
                # calculate P/R/F1/A metrics for batch
                for acc, metric in zip((precision, recall, f1, accuracy), 
                                    (precision_score, recall_score, f1_score, accuracy_score)):
                    acc.append(
                        util.calculate_metric(metric, y.cpu(), predicted_classes.cpu())
                    )
            
        
        train_loss = total_loss/batches
        val_loss = val_losses/val_batches
        print(f"Epoch {epoch+1}/{epochs}, training loss: {train_loss}, validation loss: {val_loss}")
        if not args.quiet_scores:
            util.print_scores(precision, recall, f1, accuracy, val_batches)

        # for plotting learning curve
        losses.append(train_loss) 
        metrics = util.append_metrics(precision, recall, f1, accuracy, val_batches, metrics)
        
        # Update tensorboard
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('valid_loss', val_loss, epoch)
        writer.add_scalar('valid_f1', sum(f1)/val_batches, epoch)
        writer.add_scalar('valid_accuracy', sum(accuracy)/val_batches, epoch)
        writer.add_scalar('valid_precision', sum(precision)/val_batches, epoch)
        writer.add_scalar('valid_recall', sum(recall)/val_batches, epoch)

    print(f"Training time: {time.time()-start_ts}s")


    if args.save_model:
        if not os.path.basename(args.save_model) == args.save_model:
            # If there's a dir in the filename that DNE, create it
            d = os.path.split(args.save_model)[0]
            if not os.path.exists(d):
                os.makedirs(d)
        torch.save(model.state_dict(), args.save_model)

        # Write a csv file of the model's metrics
        df = pd.DataFrame(metrics)
        csvname = os.path.splitext(args.save_model)[0] + '.csv'
        df.to_csv(csvname, index=False)

    if args.plot:
        x_ax = range(1, epochs+1)
        plt.title('Train Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Losses')
        plt.plot(x_ax, losses)
        plt.xticks(np.arange(1, x_ax[-1], 1))
        plt.show()
        plt.savefig('resnet_mnist.png')

def parse_args():
    p = argparse.ArgumentParser(description='ResNet for single channel ims')
    p.add_argument('--batch-size', type=int, default=64, metavar='N',
                   help='input batch size for training (default: 64)')
    p.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                   help='input batch size for testing (default: 128)')
    p.add_argument('--epochs', type=int, default=10, metavar='N',
                   help='number of epochs to train (default: 10)')
    p.add_argument('--seed', type=int, default=1, metavar='S',
                   help='random seed (default: 1)')
    p.add_argument('--save-model', type=str, default='models/resnet_mnist.pt',
                   help='Filename to save model to (default: models/resnet_mnist.pt')
    p.add_argument('--plot', action='store_true', default=False, 
                   help='Plot learning curve')
    p.add_argument('--quiet-scores', action='store_true', default=False,
                   help='Suppress printing of metrics after each epoch')
    p.add_argument('--load', type=str, metavar='PYTORCH_STATE_DICT',
                   help="Path to a MnistResNet model state_dict to load weights from")
    p.add_argument('--ignore-warnings', action='store_true', default=False,
                   help='Suppress printing warnings to stdout')
    p.add_argument('--tensorboard', '-tb', type=str, metavar='LOG_DIR',
                   help='tensorboard log dir')
    p.add_argument('--train-ims', type=str, metavar='PICKLE_FILE',
                   default='data/input/train_images.pkl',
                   help='Training images pkl to load (default data/input/train_images.pkl')
    p.add_argument('--depth', type=int, default=18, choices=[18, 50, 101, 152],
                   help='ResNet depth. (default: 18)')
    return p.parse_args()

def select_resnet(depth):
    if depth == 18:
        return Nets.MnistResNet()
    elif depth == 50:
        return Nets.MnistResNet50()
    elif depth == 101:
        return Nets.MnistResNet101()
    elif depth == 152:
        return Nets.MnistResNet152()
    else:
        print("WARN: Unsupported arg for ResNet depth. Returning None.")
        return None

if __name__ == '__main__':
    main()
