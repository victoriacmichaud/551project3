import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import cv2
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split

train_labels = pd.read_csv('./train_labels.csv')
train_images = pd.read_pickle('./train_images.pkl')
test_images = pd.read_pickle('test_images.pkl')
#print(train_labels.shape)
print(train_images.shape)
#print(train_images[0])

img_idx = 16
plt.title('Label: {}'.format(train_labels.iloc[img_idx]['Category']))
plt.imshow(train_images[img_idx])
#plt.show()

"""Preprocessing using OpenCV"""

def preprocessTrain():
	for i, img in enumerate(train_images):
		img = train_images[i]

		#Threshold the image for values of 0 or 1
		ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

		#Get contours of the binary image
		contoured, ctrs, heir = cv2.findContours(thresh,1,cv2.CHAIN_APPROX_SIMPLE)
		
	

preprocessTrain()