import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
import time

train_labels = pd.read_csv('./train_labels.csv')
train_images = pd.read_pickle('./train_rect_1.pkl')
test_images = pd.read_pickle('test_rect_1.pkl')

X, Y, testX, testY = [], [], [], []

Y = np.squeeze(train_labels[:80000])
testY = np.squeeze(train_labels[80000:])
X = train_images[:80000]
testX = train_images[80000:]

svm = LinearSVC()
svm.fit(X, Y) #train the SVM

svm.score(testX, testY)
