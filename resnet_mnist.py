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
import os

# Local imports
import util
from MnistResNet import MnistResNet


def main():
    args = parse_args()

    # Seed for repeatability
    torch.manual_seed(args.seed)


    start_ts = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Init model and move to gpu, if possible
    model = MnistResNet().to(device)

    # Get the DataLoaders for train and validation (test) sets
    train_params = {'batch_size': args.batch_size, 'shuffle': True}
    test_params = {'batch_size': args.test_batch_size, 'shuffle': False}
    train_loader, val_loader = util.data_loaders(test_params, train_params)

    # your loss function, cross entropy works well for multi-class problems
    loss_function = nn.CrossEntropyLoss() 

    # optimizer, I've used Adadelta, as it works well without any magic numbers
    optimizer = optim.Adadelta(model.parameters())

    losses = []
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
            progress.set_description("Loss: {:.4f}".format(total_loss/(i+1)))
            
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
            
        
        if not args.quiet_scores:
            print(f"Epoch {epoch+1}/{epochs}, training loss: {total_loss/batches}, validation loss: {val_losses/val_batches}")
            util.print_scores(precision, recall, f1, accuracy, val_batches)
        losses.append(total_loss/batches) # for plotting learning curve
    print(f"Training time: {time.time()-start_ts}s")

    if args.save_model:
        if not os.path.basename(args.save_name) == args.save_name:
            # If there's a dir in the filename that DNE, create it
            d = os.path.split(args.save_name)[0]
            if not os.path.exists(d):
                os.makedirs(d)
        torch.save(model.state_dict(), args.save_name)

    if args.plot:
        x_ax = range(1, epochs+1)
        plt.xlabel('Epochs')
        plt.ylabel('Losses')
        plt.plot(x_ax, losses)
        plt.xticks(np.arange(1, x_ax[-1], 1))
        plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description='ResNet for single channel ims')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--test-batch-size', type=int, default=500, metavar='N',
                        help='input batch size for testing (default: 500)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    # parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
    #                     help='learning rate (default: 0.01)')
    # parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
    #                     help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model. Will overwrite files')
    parser.add_argument('--save-name', type=str, default='models/resnet_mnist.pt',
                        help='Filename to save model to (default: models/resnet_mnist.pt')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='Plot learning curve')
    parser.add_argument('--quiet-scores', action='store_true', default=False,
                        help='Suppress printing of metrics after each epoch')
    return parser.parse_args()

if __name__ == '__main__':
    main()
