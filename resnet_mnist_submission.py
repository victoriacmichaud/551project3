#!/usr/bin/env python3

import torch
import argparse
import numpy as np
import os
import pandas as pd
from datetime import date

# Local imports
import util
from MnistResNet import MnistResNet


def write_submission(outfile, Y_pred):
    """ Write Id,Category csv for submission to kaggle
    Y_pred is a 1d numpy array containing the predicted classes
    """
    with open(outfile, 'w') as f:
        f.write('Id,Category\n')
        for i in range(len(Y_pred)):
            f.write(str(int(i)))
            f.write(',')
            f.write(str(int(Y_pred[i])))
            f.write('\n')

parser = argparse.ArgumentParser(description='Write submission for ResNet for single channel ims')
parser.add_argument('model_weights', type=str, help='MnistResNet weights to load')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model
model = MnistResNet().to(device)
model.load_state_dict(torch.load(args.model_weights))

# Load data
# This is using another Dataset class with the same transform applied and a
# batch size equal to the length of the dataset
X_loader = util.load_submission_data()

# Get predicted classes
with torch.no_grad():
    for i, X in enumerate(X_loader):
        X  = X.to(device)
        outputs = model(X) # get prediction from network
        ypred = torch.max(outputs, 1)[1] # get class from network's prediction

# Write a submission csv with columns Id, Category
ypred = ypred.to('cpu')
outname = os.path.basename(args.model_weights)
outname = os.path.splitext(outname)[0] + '.csv'
outname = str(date.today()) + '_submission-' + outname
outfile = os.path.join('data/output', outname)
print('Writing submission csv for %s to %s' %(args.model_weights, outfile))
write_submission(outfile, ypred)

