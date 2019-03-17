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
test_images = pd.read_pickle('./test_rect_1.pkl')

# X, Y, testX, testY = [], [], [], []

# Y = np.squeeze(train_labels[:80000])
# testY = np.squeeze(train_labels[80000:])
# X = train_images[:80000]
# testX = train_images[80000:]


labels_formatted = train_labels.as_matrix(columns=train_labels.columns[1:])

X, testX, Y, testY = train_test_split(train_images, labels_formatted, test_size = 0.2)
print("Array shapes: ")
print(np.shape(X), np.shape(testX), np.shape(Y), np.shape(testY))


nsamples, nX, nY = np.shape(X)
X = np.reshape(X, (nsamples, nX*nY))
Y = np.reshape(Y, (np.shape(Y)[0], ))
print("Reshaped: ", np.shape(X), np.shape(Y))

# #Implement SVM
svm = LinearSVC(verbose=3,max_iter=10000)
svm.fit(X, Y) #train the SVM

predictY = svm.predict(testX)

# # Model accuracy
print("Score: ")
svm.score(testX, testY)
print("Accuracy: ")
metrics.accuracy_score(testY, predictY)
