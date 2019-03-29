#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import time

# Nick
import pickle
import os
from os.path import join as pj
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

# Local
from util import plot_confusion_matrix


def main():
    """ Train a lsvc on the raw and each type of preprocessed images and save:
        - a model pickle
        - metrics report csv
        - normalized confusion matrix plot
    """

    # My ims were in 'data/input', just set this to '' if yours are flat in
    # your working dir
    input_dir = 'data/input'
    train_labels = pd.read_csv(pj(input_dir, 'train_labels.csv'))

    suffixes = ['images', 'pix_1', 'ellipse_1', 'rect_1']
    for s in suffixes:
        f = '_'.join(('train', s +'.pkl')) 
        train_images = pd.read_pickle(pj(input_dir, f))

        model_name = '-'.join(('lsvc', s))
        clf = LinearSVC(tol=1e-7, verbose=1)
        run_model(train_images, train_labels, clf, model_name,
                  save_dir='data/output/svm')
    
    print('YOU ARE ALL FREE NOW')

def run_model(train_images, train_labels, clf, model_name, save_dir='',
              show_plot=False):
    """Train a model and save some of its outputs
    train_images: 3d np.array, result of reading train_images pickle
    train_labels: pd.DataFrame, result of pd.read_csv
    clf: the sklearn linear classifier to use
    model_name: str, string to prepend saved items with 
    save_dir: optional dir to save into
    show_plot: show the confusion plot before saving it

    Saved:
        model_name.pkl : pickled model
        model_name-report.csv : classification report
        model_name-confusion_matrix.png : confusion matrix plot
    """
    print('\nRunning model', model_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_name = pj(save_dir, model_name)
    print('Saving model and outputs to %s*' % save_dir)

    clf, ypred, testY, testX = train_clf(train_images, train_labels, clf)
    save_pickle(model_name +'.pkl', clf)

    report = classification_report(testY, ypred, output_dict=True)
    df = pd.DataFrame(report)
    df.to_csv(model_name +'-report.csv')

    plot_confusion_matrix(testY, ypred, clf.classes_, normalize=True)
    plt.savefig(model_name + '-confusion_matrix.png', dpi=80)
    if show_plot:
        plt.show()

def train_clf(train_images, train_labels, clf):
    """ return clf, ypred, testY, testX
    Train a linear classifier on supplied images. 
    """

    # Remove column labels. Shape (N, 1)
    labels_formatted = train_labels.as_matrix(columns=train_labels.columns[1:])

    # Reshape data to 2d arrays
    N, nX, nY = np.shape(train_images)
    train_images = np.reshape(train_images, (N, nX*nY))
    labels_formatted = np.reshape(labels_formatted, 
                                  np.shape(labels_formatted)[0], )

    # Normalize data
    train_images = normalize(train_images)

    # Add bias term
    train_images = add_bias(train_images)

    X, testX, Y, testY = train_test_split(train_images, labels_formatted, test_size = 0.2)

    # Reshape data to 2d arrays
    # nsamples, nX, nY = np.shape(X)
    # X = np.reshape(X, (nsamples, nX*nY))
    Y = np.reshape(Y, (np.shape(Y)[0], ))
    testY = np.reshape(testY, (np.shape(testY)[0], ))

    # Fit Clf
    print('Fitting clf')
    clf.fit(X, Y) #train the clf
    print('Getting clf predictions')
    predictY = clf.predict(testX)

    # Clf Metrics 
    print("Accuracy:", clf.score(testX, testY))
    # print("Accuracy:", metrics.accuracy_score(testY, predictY))

    # dbg 
    return clf, predictY, testY, testX

def add_bias(X, val=1):
    """Add a constant bias term to each row in X (a 2d np array)"""
    tmp = [np.append(x, val) for x in X]
    return np.array(tmp)

def save_pickle(filename, obj):
    """ Save a python object to disk
    pickle files usually have a .p or .pickle extension
    """
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

if __name__ == '__main__':
    main()
