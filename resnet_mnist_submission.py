#!/usr/bin/env python3

import torch
import argparse
import numpy as np
import os
import pandas as pd
from datetime import date

# Local imports
import util
import MnistResNet as Nets

def main():
    p = argparse.ArgumentParser(description='Write Kaggle submission with a loaded resnet')
    p.add_argument('model_weights', type=str, help='MnistResNet weights to load')
    p.add_argument('--depth', type=int, default=18, choices=[18, 50, 101, 152],
                   help='ResNet depth. Must match the loaded weights (default: 18)')
    p.add_argument('--batch_size', type=int, default=128, 
                   help='DatLoader batch size (default: 128)')
    args = p.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model
    model = select_resnet(args.depth)
    model.to(device)
    model.load_state_dict(torch.load(args.model_weights))

    # Load data
    # This is using another Dataset class with the same transform applied but
    # loading the test (submission) data 
    X_loader = util.submission_data_loader(args.batch_size)

    # Get predicted classes
    # Deep models will need predictions to be calculated in batches, hence the
    # list and hstack
    ypred = []
    with torch.no_grad():
        for i, X in enumerate(X_loader):
            X  = X.to(device)
            outputs = model(X) # get prediction from network
            ypred.append(torch.max(outputs, 1)[1]) # get class from prediction
    ypred = [y.to('cpu') for y in ypred]
    ypred = np.hstack(ypred)

    # Write a submission csv with columns Id, Category
    outname = os.path.basename(args.model_weights)
    outname = os.path.splitext(outname)[0] + '.csv'
    outname = str(date.today()) + '_submission-' + outname
    outfile = os.path.join('data/output', outname)
    print('Writing submission csv for %s to %s' %(args.model_weights, outfile))
    write_submission(outfile, ypred)

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

if __name__ == '__main__':
    main()
